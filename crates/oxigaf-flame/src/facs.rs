//! Facial Action Coding System (FACS) mapping to FLAME expression parameters.
//!
//! This module provides a mapping between standard FACS Action Units (AUs) and
//! the FLAME parametric head model's expression basis coefficients.  The mapping
//! is approximate and research-based; it allows generating plausible facial
//! expressions by specifying AU intensities rather than raw FLAME parameters.
//!
//! ## Quick Start
//!
//! ```rust
//! use oxigaf_flame::facs::{FacsToFlame, FacsPresets};
//!
//! // Create a converter with default FLAME mappings
//! let converter = FacsToFlame::default();
//!
//! // Use a preset expression
//! let smile = FacsPresets::smile();
//! let expr_deltas = converter.apply_aus(&smile);
//! println!("Smile expression has {} coefficients", expr_deltas.len());
//! ```

use crate::params::FlameParams;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ActionUnit enum
// ---------------------------------------------------------------------------

/// FACS Action Unit identifiers.
///
/// Each variant corresponds to one of the standard FACS-defined muscle
/// movements.  `Custom(u32)` allows application-specific extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionUnit {
    /// AU01 – Inner Brow Raise
    InnerBrowRaise,
    /// AU02 – Outer Brow Raise
    OuterBrowRaise,
    /// AU04 – Brow Lowerer
    BrowLowerer,
    /// AU05 – Upper Lid Raiser
    UpperLidRaiser,
    /// AU06 – Cheek Raiser
    CheekRaiser,
    /// AU07 – Lid Tightener
    LidTightener,
    /// AU09 – Nose Wrinkler
    NoseWrinkler,
    /// AU10 – Upper Lip Raiser
    UpperLipRaiser,
    /// AU12 – Lip Corner Puller (smile)
    LipCornerPuller,
    /// AU13 – Cheek Puffer
    CheekPuffer,
    /// AU14 – Dimpler
    Dimpler,
    /// AU15 – Lip Corner Depressor
    LipCornerDepressor,
    /// AU17 – Chin Raiser
    ChinRaiser,
    /// AU20 – Lip Stretcher
    LipStretcher,
    /// AU23 – Lip Tightener
    LipTightener,
    /// AU24 – Lip Pressor
    LipPressor,
    /// AU25 – Lips Part
    LipsPart,
    /// AU26 – Jaw Drop
    JawDrop,
    /// AU28 – Lip Suck
    LipSuck,
    /// AU43 – Eyes Closed (blink/close)
    EyesClosed,
    /// AU45 – Blink
    Blink,
    /// Custom action unit by numeric ID.
    Custom(u32),
}

/// Static slice of all 21 standard (non-Custom) action units.
static ALL_STANDARD_AUS: &[ActionUnit] = &[
    ActionUnit::InnerBrowRaise,
    ActionUnit::OuterBrowRaise,
    ActionUnit::BrowLowerer,
    ActionUnit::UpperLidRaiser,
    ActionUnit::CheekRaiser,
    ActionUnit::LidTightener,
    ActionUnit::NoseWrinkler,
    ActionUnit::UpperLipRaiser,
    ActionUnit::LipCornerPuller,
    ActionUnit::CheekPuffer,
    ActionUnit::Dimpler,
    ActionUnit::LipCornerDepressor,
    ActionUnit::ChinRaiser,
    ActionUnit::LipStretcher,
    ActionUnit::LipTightener,
    ActionUnit::LipPressor,
    ActionUnit::LipsPart,
    ActionUnit::JawDrop,
    ActionUnit::LipSuck,
    ActionUnit::EyesClosed,
    ActionUnit::Blink,
];

impl ActionUnit {
    /// Human-readable `snake_case` name for this action unit.
    ///
    /// For `Custom(_)` variants, always returns `"custom"`.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::InnerBrowRaise => "inner_brow_raise",
            Self::OuterBrowRaise => "outer_brow_raise",
            Self::BrowLowerer => "brow_lowerer",
            Self::UpperLidRaiser => "upper_lid_raiser",
            Self::CheekRaiser => "cheek_raiser",
            Self::LidTightener => "lid_tightener",
            Self::NoseWrinkler => "nose_wrinkler",
            Self::UpperLipRaiser => "upper_lip_raiser",
            Self::LipCornerPuller => "lip_corner_puller",
            Self::CheekPuffer => "cheek_puffer",
            Self::Dimpler => "dimpler",
            Self::LipCornerDepressor => "lip_corner_depressor",
            Self::ChinRaiser => "chin_raiser",
            Self::LipStretcher => "lip_stretcher",
            Self::LipTightener => "lip_tightener",
            Self::LipPressor => "lip_pressor",
            Self::LipsPart => "lips_part",
            Self::JawDrop => "jaw_drop",
            Self::LipSuck => "lip_suck",
            Self::EyesClosed => "eyes_closed",
            Self::Blink => "blink",
            Self::Custom(_) => "custom",
        }
    }

    /// Numeric AU identifier.
    ///
    /// For `Custom(_)` variants, returns `u32::MAX` (the numeric ID is stored
    /// in the variant itself but is inaccessible through this method).
    #[must_use]
    pub fn au_number(&self) -> u32 {
        match self {
            Self::InnerBrowRaise => 1,
            Self::OuterBrowRaise => 2,
            Self::BrowLowerer => 4,
            Self::UpperLidRaiser => 5,
            Self::CheekRaiser => 6,
            Self::LidTightener => 7,
            Self::NoseWrinkler => 9,
            Self::UpperLipRaiser => 10,
            Self::LipCornerPuller => 12,
            Self::CheekPuffer => 13,
            Self::Dimpler => 14,
            Self::LipCornerDepressor => 15,
            Self::ChinRaiser => 17,
            Self::LipStretcher => 20,
            Self::LipTightener => 23,
            Self::LipPressor => 24,
            Self::LipsPart => 25,
            Self::JawDrop => 26,
            Self::LipSuck => 28,
            Self::EyesClosed => 43,
            Self::Blink => 45,
            Self::Custom(_) => u32::MAX,
        }
    }

    /// All 21 standard action units (excludes `Custom` variants).
    #[must_use]
    pub fn all_standard() -> &'static [ActionUnit] {
        ALL_STANDARD_AUS
    }
}

// ---------------------------------------------------------------------------
// FacsIntensity
// ---------------------------------------------------------------------------

/// Intensity of an action unit activation, clamped to `[0.0, 1.0]`.
///
/// A value of `0.0` means the AU is inactive (neutral), and `1.0` means
/// maximum activation.
#[derive(Debug, Clone, Copy)]
pub struct FacsIntensity(pub f32);

impl FacsIntensity {
    /// Create a new intensity, clamping the value to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    /// Neutral (zero) intensity.
    #[must_use]
    pub fn neutral() -> Self {
        Self(0.0)
    }

    /// Maximum (full) intensity.
    #[must_use]
    pub fn max() -> Self {
        Self(1.0)
    }

    /// Raw intensity value in `[0.0, 1.0]`.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// AuMapping
// ---------------------------------------------------------------------------

/// Maps one `ActionUnit` activation to expression parameter deltas.
///
/// At intensity `I`, for each `(param_idx, scale)` pair:
/// ```text
/// expression[param_idx] += scale * I
/// ```
#[derive(Debug, Clone)]
pub struct AuMapping {
    /// The action unit this mapping applies to.
    pub action_unit: ActionUnit,
    /// `(expression_param_index, scale_at_full_intensity)` pairs.
    pub contributions: Vec<(usize, f32)>,
    /// Minimum number of expression dimensions required (i.e. `max_index + 1`).
    pub required_expr_dims: usize,
}

impl AuMapping {
    /// Create a new `AuMapping`.
    ///
    /// `required_expr_dims` is derived automatically as `max(index) + 1` over
    /// all contribution pairs, or `0` if `contributions` is empty.
    #[must_use]
    pub fn new(au: ActionUnit, contributions: Vec<(usize, f32)>) -> Self {
        let required = contributions.iter().map(|(i, _)| i + 1).max().unwrap_or(0);
        Self {
            action_unit: au,
            contributions,
            required_expr_dims: required,
        }
    }
}

// ---------------------------------------------------------------------------
// FacsLibrary
// ---------------------------------------------------------------------------

/// A library of `ActionUnit` → `AuMapping` associations.
///
/// Provides default FLAME-AU mappings based on published research (approximate).
pub struct FacsLibrary {
    mappings: HashMap<ActionUnit, AuMapping>,
}

impl FacsLibrary {
    /// Build the default FLAME-AU library.
    ///
    /// The mappings are approximate research-based associations between
    /// standard FACS action units and FLAME expression basis indices.
    /// They are plausible but not ground-truth calibrated.
    #[must_use]
    pub fn default_flame() -> Self {
        let entries: Vec<AuMapping> = vec![
            // AU01 InnerBrowRaise → expr[0] += 1.2
            AuMapping::new(ActionUnit::InnerBrowRaise, vec![(0, 1.2)]),
            // AU02 OuterBrowRaise → expr[1] += 1.0, expr[2] += 0.3
            AuMapping::new(ActionUnit::OuterBrowRaise, vec![(1, 1.0), (2, 0.3)]),
            // AU04 BrowLowerer    → expr[3] += -1.0, expr[0] += -0.5
            AuMapping::new(ActionUnit::BrowLowerer, vec![(3, -1.0), (0, -0.5)]),
            // AU05 UpperLidRaiser → expr[4] += 0.8
            AuMapping::new(ActionUnit::UpperLidRaiser, vec![(4, 0.8)]),
            // AU06 CheekRaiser    → expr[5] += 0.6
            AuMapping::new(ActionUnit::CheekRaiser, vec![(5, 0.6)]),
            // AU07 LidTightener   → expr[4] += -0.3
            AuMapping::new(ActionUnit::LidTightener, vec![(4, -0.3)]),
            // AU09 NoseWrinkler   → expr[6] += 0.7
            AuMapping::new(ActionUnit::NoseWrinkler, vec![(6, 0.7)]),
            // AU10 UpperLipRaiser → expr[7] += 0.5
            AuMapping::new(ActionUnit::UpperLipRaiser, vec![(7, 0.5)]),
            // AU12 LipCornerPuller→ expr[8] += 1.5, expr[9] += 0.3
            AuMapping::new(ActionUnit::LipCornerPuller, vec![(8, 1.5), (9, 0.3)]),
            // AU13 CheekPuffer    → expr[5] += 0.4, expr[10] += 0.5
            AuMapping::new(ActionUnit::CheekPuffer, vec![(5, 0.4), (10, 0.5)]),
            // AU14 Dimpler        → expr[11] += 0.3
            AuMapping::new(ActionUnit::Dimpler, vec![(11, 0.3)]),
            // AU15 LipCornerDep.  → expr[8] += -1.0, expr[12] += 0.7
            AuMapping::new(ActionUnit::LipCornerDepressor, vec![(8, -1.0), (12, 0.7)]),
            // AU17 ChinRaiser     → expr[13] += 0.4
            AuMapping::new(ActionUnit::ChinRaiser, vec![(13, 0.4)]),
            // AU20 LipStretcher   → expr[14] += 0.8
            AuMapping::new(ActionUnit::LipStretcher, vec![(14, 0.8)]),
            // AU23 LipTightener   → expr[15] += 0.5
            AuMapping::new(ActionUnit::LipTightener, vec![(15, 0.5)]),
            // AU24 LipPressor     → expr[16] += 0.4
            AuMapping::new(ActionUnit::LipPressor, vec![(16, 0.4)]),
            // AU25 LipsPart       → expr[17] += 1.2
            AuMapping::new(ActionUnit::LipsPart, vec![(17, 1.2)]),
            // AU26 JawDrop        → expr[17] += 0.8, expr[18] += 1.0 (jaw angle)
            AuMapping::new(ActionUnit::JawDrop, vec![(17, 0.8), (18, 1.0)]),
            // AU28 LipSuck        → expr[19] += 0.6
            AuMapping::new(ActionUnit::LipSuck, vec![(19, 0.6)]),
            // AU43 EyesClosed     → expr[20] += 1.0, expr[21] += 1.0
            AuMapping::new(ActionUnit::EyesClosed, vec![(20, 1.0), (21, 1.0)]),
            // AU45 Blink          → expr[20] += 0.8, expr[21] += 0.8
            AuMapping::new(ActionUnit::Blink, vec![(20, 0.8), (21, 0.8)]),
        ];

        let mut mappings = HashMap::with_capacity(entries.len());
        for m in entries {
            mappings.insert(m.action_unit, m);
        }
        Self { mappings }
    }

    /// Look up the mapping for an action unit.
    #[must_use]
    pub fn get_mapping(&self, au: &ActionUnit) -> Option<&AuMapping> {
        self.mappings.get(au)
    }

    /// Insert or replace a mapping.
    pub fn add_mapping(&mut self, mapping: AuMapping) {
        self.mappings.insert(mapping.action_unit, mapping);
    }

    /// Number of registered mappings.
    #[must_use]
    pub fn num_mappings(&self) -> usize {
        self.mappings.len()
    }

    /// Collect all action units that have a mapping.
    #[must_use]
    pub fn all_mapped_aus(&self) -> Vec<ActionUnit> {
        self.mappings.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// FacsToFlame
// ---------------------------------------------------------------------------

/// Converts FACS AU activations into FLAME expression parameter deltas.
pub struct FacsToFlame {
    library: FacsLibrary,
    /// Number of expression parameters in the output vector (default: 50).
    pub expr_dims: usize,
}

impl FacsToFlame {
    /// Create a converter with an explicit library and expression dimensionality.
    #[must_use]
    pub fn new(library: FacsLibrary, expr_dims: usize) -> Self {
        Self { library, expr_dims }
    }

    /// Convert a set of AU activations to expression parameter deltas.
    ///
    /// Returns a `Vec<f32>` of length `expr_dims`.  Contributions from all
    /// activated AUs are accumulated, then each element is clamped to
    /// `[-3.0, 3.0]` to prevent extreme deformations.  AUs without a
    /// registered mapping are silently ignored.
    #[must_use]
    pub fn apply_aus(&self, activations: &HashMap<ActionUnit, FacsIntensity>) -> Vec<f32> {
        let mut expr = vec![0.0f32; self.expr_dims];

        for (au, intensity) in activations {
            if let Some(mapping) = self.library.get_mapping(au) {
                let i = intensity.value();
                for &(param_idx, scale) in &mapping.contributions {
                    if param_idx < self.expr_dims {
                        expr[param_idx] += scale * i;
                    }
                }
            }
        }

        // Clamp to prevent extreme deformations.
        for v in &mut expr {
            *v = v.clamp(-3.0, 3.0);
        }

        expr
    }

    /// Apply AUs and return a full `FlameParams` with the computed expression.
    ///
    /// Shape, pose, and translation are set to neutral/zero values.
    #[must_use]
    pub fn to_flame_params(&self, activations: &HashMap<ActionUnit, FacsIntensity>) -> FlameParams {
        let expression = self.apply_aus(activations);
        FlameParams {
            shape: Vec::new(),
            expression,
            pose: vec![0.0; FlameParams::NUM_JOINTS * 3],
            translation: [0.0; 3],
        }
    }

    /// Validate AU activations against the library.
    ///
    /// Returns a list of AUs present in `activations` that have no registered
    /// mapping.  An empty return value means all AUs are mapped.
    #[must_use]
    pub fn validate_aus(
        &self,
        activations: &HashMap<ActionUnit, FacsIntensity>,
    ) -> Vec<ActionUnit> {
        activations
            .keys()
            .filter(|au| self.library.get_mapping(au).is_none())
            .copied()
            .collect()
    }

    /// Compute the expression parameter vector for a single AU at the given
    /// intensity.  Intensity is clamped to `[0.0, 1.0]` before use.
    #[must_use]
    pub fn single_au_expression(&self, au: ActionUnit, intensity: f32) -> Vec<f32> {
        let mut activations = HashMap::with_capacity(1);
        activations.insert(au, FacsIntensity::new(intensity));
        self.apply_aus(&activations)
    }
}

impl Default for FacsToFlame {
    /// Create a default converter using `FacsLibrary::default_flame()` and 50
    /// expression dimensions.
    fn default() -> Self {
        Self::new(FacsLibrary::default_flame(), 50)
    }
}

// ---------------------------------------------------------------------------
// FacsPresets
// ---------------------------------------------------------------------------

/// Preset FACS expressions composed from standard AU combinations.
///
/// These presets are commonly used in affective computing and animation
/// research; intensities are chosen for natural-looking results.
pub struct FacsPresets;

impl FacsPresets {
    /// Neutral face: no AUs activated.
    #[must_use]
    pub fn neutral() -> HashMap<ActionUnit, FacsIntensity> {
        HashMap::new()
    }

    /// Smile: cheek raiser (AU06) + lip corner puller (AU12) at 0.8.
    #[must_use]
    pub fn smile() -> HashMap<ActionUnit, FacsIntensity> {
        let mut m = HashMap::new();
        m.insert(ActionUnit::CheekRaiser, FacsIntensity::new(0.8));
        m.insert(ActionUnit::LipCornerPuller, FacsIntensity::new(0.8));
        m
    }

    /// Surprise: inner/outer brow raise + upper lid raiser + jaw drop.
    #[must_use]
    pub fn surprise() -> HashMap<ActionUnit, FacsIntensity> {
        let mut m = HashMap::new();
        m.insert(ActionUnit::InnerBrowRaise, FacsIntensity::new(0.9));
        m.insert(ActionUnit::OuterBrowRaise, FacsIntensity::new(0.9));
        m.insert(ActionUnit::UpperLidRaiser, FacsIntensity::new(0.7));
        m.insert(ActionUnit::JawDrop, FacsIntensity::new(0.6));
        m
    }

    /// Angry: brow lowerer + lid tightener + lip tightener + lip pressor.
    #[must_use]
    pub fn angry() -> HashMap<ActionUnit, FacsIntensity> {
        let mut m = HashMap::new();
        m.insert(ActionUnit::BrowLowerer, FacsIntensity::new(0.8));
        m.insert(ActionUnit::LidTightener, FacsIntensity::new(0.6));
        m.insert(ActionUnit::LipTightener, FacsIntensity::new(0.5));
        m.insert(ActionUnit::LipPressor, FacsIntensity::new(0.4));
        m
    }

    /// Sad: inner brow raise + brow lowerer + lip corner depressor + chin raiser.
    #[must_use]
    pub fn sad() -> HashMap<ActionUnit, FacsIntensity> {
        let mut m = HashMap::new();
        m.insert(ActionUnit::InnerBrowRaise, FacsIntensity::new(0.6));
        m.insert(ActionUnit::BrowLowerer, FacsIntensity::new(0.4));
        m.insert(ActionUnit::LipCornerDepressor, FacsIntensity::new(0.7));
        m.insert(ActionUnit::ChinRaiser, FacsIntensity::new(0.5));
        m
    }

    /// Disgust: nose wrinkler + lip corner depressor + lips part (approximate).
    #[must_use]
    pub fn disgust() -> HashMap<ActionUnit, FacsIntensity> {
        let mut m = HashMap::new();
        m.insert(ActionUnit::NoseWrinkler, FacsIntensity::new(0.8));
        m.insert(ActionUnit::LipCornerDepressor, FacsIntensity::new(0.5));
        m.insert(ActionUnit::LipsPart, FacsIntensity::new(0.3));
        m
    }

    /// Fear: brow raise + brow lowerer + lid raiser/tightener + lip stretcher + jaw drop.
    #[must_use]
    pub fn fear() -> HashMap<ActionUnit, FacsIntensity> {
        let mut m = HashMap::new();
        m.insert(ActionUnit::InnerBrowRaise, FacsIntensity::new(0.7));
        m.insert(ActionUnit::OuterBrowRaise, FacsIntensity::new(0.7));
        m.insert(ActionUnit::BrowLowerer, FacsIntensity::new(0.4));
        m.insert(ActionUnit::UpperLidRaiser, FacsIntensity::new(0.8));
        m.insert(ActionUnit::LidTightener, FacsIntensity::new(0.5));
        m.insert(ActionUnit::LipStretcher, FacsIntensity::new(0.6));
        m.insert(ActionUnit::JawDrop, FacsIntensity::new(0.4));
        m
    }

    /// Blink: AU45 at full intensity.
    #[must_use]
    pub fn blink() -> HashMap<ActionUnit, FacsIntensity> {
        let mut m = HashMap::new();
        m.insert(ActionUnit::Blink, FacsIntensity::new(1.0));
        m
    }

    /// Names of all available presets.
    #[must_use]
    pub fn preset_names() -> &'static [&'static str] {
        &[
            "neutral", "smile", "surprise", "angry", "sad", "disgust", "fear", "blink",
        ]
    }

    /// Look up a preset by name (case-insensitive).
    ///
    /// Returns `None` if the name does not match any known preset.
    #[must_use]
    pub fn by_name(name: &str) -> Option<HashMap<ActionUnit, FacsIntensity>> {
        match name.to_lowercase().as_str() {
            "neutral" => Some(Self::neutral()),
            "smile" => Some(Self::smile()),
            "surprise" => Some(Self::surprise()),
            "angry" => Some(Self::angry()),
            "sad" => Some(Self::sad()),
            "disgust" => Some(Self::disgust()),
            "fear" => Some(Self::fear()),
            "blink" => Some(Self::blink()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ActionUnit -------------------------------------------------------

    #[test]
    fn test_action_unit_name() {
        assert_eq!(ActionUnit::InnerBrowRaise.name(), "inner_brow_raise");
        assert_eq!(ActionUnit::LipCornerPuller.name(), "lip_corner_puller");
        assert_eq!(ActionUnit::Blink.name(), "blink");
        assert_eq!(ActionUnit::Custom(99).name(), "custom");
    }

    #[test]
    fn test_action_unit_au_number() {
        assert_eq!(ActionUnit::InnerBrowRaise.au_number(), 1);
        assert_eq!(ActionUnit::OuterBrowRaise.au_number(), 2);
        assert_eq!(ActionUnit::BrowLowerer.au_number(), 4);
        assert_eq!(ActionUnit::JawDrop.au_number(), 26);
        assert_eq!(ActionUnit::EyesClosed.au_number(), 43);
        assert_eq!(ActionUnit::Blink.au_number(), 45);
        assert_eq!(ActionUnit::Custom(7).au_number(), u32::MAX);
    }

    #[test]
    fn test_all_standard_aus_count() {
        assert_eq!(ActionUnit::all_standard().len(), 21);
    }

    #[test]
    fn test_all_standard_aus_no_custom() {
        for au in ActionUnit::all_standard() {
            assert!(
                !matches!(au, ActionUnit::Custom(_)),
                "all_standard should not contain Custom variants"
            );
        }
    }

    // ---- FacsIntensity ----------------------------------------------------

    #[test]
    fn test_facs_intensity_clamp() {
        assert_eq!(FacsIntensity::new(1.5).value(), 1.0);
        assert_eq!(FacsIntensity::new(-0.5).value(), 0.0);
        assert!((FacsIntensity::new(0.6).value() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_facs_intensity_neutral_max() {
        assert_eq!(FacsIntensity::neutral().value(), 0.0);
        assert_eq!(FacsIntensity::max().value(), 1.0);
    }

    // ---- AuMapping --------------------------------------------------------

    #[test]
    fn test_au_mapping_required_dims() {
        let m = AuMapping::new(ActionUnit::Blink, vec![(5, 0.5), (12, 0.8)]);
        assert_eq!(m.required_expr_dims, 13); // max index 12 → 12+1

        let empty = AuMapping::new(ActionUnit::Dimpler, vec![]);
        assert_eq!(empty.required_expr_dims, 0);
    }

    #[test]
    fn test_au_mapping_stores_contributions() {
        let contribs = vec![(0, 1.2_f32), (3, -0.5_f32)];
        let m = AuMapping::new(ActionUnit::InnerBrowRaise, contribs.clone());
        assert_eq!(m.contributions.len(), 2);
        assert_eq!(m.action_unit, ActionUnit::InnerBrowRaise);
    }

    // ---- FacsLibrary ------------------------------------------------------

    #[test]
    fn test_library_default_flame_has_all_standard_aus() {
        let lib = FacsLibrary::default_flame();
        for au in ActionUnit::all_standard() {
            assert!(
                lib.get_mapping(au).is_some(),
                "Missing mapping for {:?} (AU{})",
                au,
                au.au_number()
            );
        }
    }

    #[test]
    fn test_library_add_custom_mapping() {
        let mut lib = FacsLibrary::default_flame();
        let before = lib.num_mappings();
        let custom_mapping = AuMapping::new(ActionUnit::Custom(99), vec![(30, 0.5)]);
        lib.add_mapping(custom_mapping);
        assert_eq!(lib.num_mappings(), before + 1);
        assert!(lib.get_mapping(&ActionUnit::Custom(99)).is_some());
    }

    #[test]
    fn test_library_all_mapped_aus_count() {
        let lib = FacsLibrary::default_flame();
        let aus = lib.all_mapped_aus();
        assert_eq!(aus.len(), lib.num_mappings());
        // At minimum the 21 standard AUs
        assert!(aus.len() >= 21);
    }

    #[test]
    fn test_library_replace_mapping() {
        let mut lib = FacsLibrary::default_flame();
        let old_len = lib.num_mappings();
        // Re-inserting InnerBrowRaise should replace, not add.
        let new_m = AuMapping::new(ActionUnit::InnerBrowRaise, vec![(0, 2.0)]);
        lib.add_mapping(new_m);
        assert_eq!(
            lib.num_mappings(),
            old_len,
            "replace should not increase count"
        );
        let m = lib
            .get_mapping(&ActionUnit::InnerBrowRaise)
            .expect("InnerBrowRaise mapping must exist after insertion");
        assert!((m.contributions[0].1 - 2.0).abs() < 1e-6);
    }

    // ---- FacsToFlame ------------------------------------------------------

    #[test]
    fn test_facs_to_flame_params_length() {
        let converter = FacsToFlame::default();
        let activations = FacsPresets::smile();
        let expr = converter.apply_aus(&activations);
        assert_eq!(expr.len(), 50);
    }

    #[test]
    fn test_facs_to_flame_single_au() {
        let converter = FacsToFlame::default();
        let expr = converter.single_au_expression(ActionUnit::InnerBrowRaise, 1.0);
        // AU01 → expr[0] += 1.2
        assert!(
            (expr[0] - 1.2).abs() < 1e-5,
            "expr[0] should be 1.2, got {}",
            expr[0]
        );
        // Other indices should be zero for this AU
        for (i, v) in expr.iter().enumerate() {
            if i != 0 {
                assert!(*v == 0.0, "expr[{i}] should be 0, got {v}");
            }
        }
    }

    #[test]
    fn test_facs_to_flame_multiple_aus() {
        let converter = FacsToFlame::default();
        let mut activations = HashMap::new();
        // AU12 (LipCornerPuller) → expr[8] += 1.5, expr[9] += 0.3
        // AU06 (CheekRaiser)    → expr[5] += 0.6
        activations.insert(ActionUnit::LipCornerPuller, FacsIntensity::new(1.0));
        activations.insert(ActionUnit::CheekRaiser, FacsIntensity::new(1.0));
        let expr = converter.apply_aus(&activations);
        assert!((expr[8] - 1.5).abs() < 1e-5, "expr[8] = {}", expr[8]);
        assert!((expr[9] - 0.3).abs() < 1e-5, "expr[9] = {}", expr[9]);
        assert!((expr[5] - 0.6).abs() < 1e-5, "expr[5] = {}", expr[5]);
    }

    #[test]
    fn test_facs_to_flame_clamps_output() {
        // Use a custom mapping to drive a coefficient to 10.0, verifying clamp to 3.0.
        let mut lib = FacsLibrary::default_flame();
        // Add a custom AU that pushes expr[0] to 10.0
        lib.add_mapping(AuMapping::new(ActionUnit::Custom(1), vec![(0, 10.0)]));
        let converter2 = FacsToFlame::new(lib, 50);
        let mut activations = HashMap::new();
        activations.insert(ActionUnit::Custom(1), FacsIntensity::new(1.0));
        let expr = converter2.apply_aus(&activations);
        assert!(
            expr[0] <= 3.0,
            "expr[0] should be clamped to 3.0, got {}",
            expr[0]
        );
        // Also test negative clamp
        let mut lib2 = FacsLibrary::default_flame();
        lib2.add_mapping(AuMapping::new(ActionUnit::Custom(2), vec![(1, -10.0)]));
        let converter3 = FacsToFlame::new(lib2, 50);
        let mut acts2 = HashMap::new();
        acts2.insert(ActionUnit::Custom(2), FacsIntensity::new(1.0));
        let expr2 = converter3.apply_aus(&acts2);
        assert!(
            expr2[1] >= -3.0,
            "expr[1] should be clamped to -3.0, got {}",
            expr2[1]
        );
    }

    #[test]
    fn test_facs_to_flame_params_struct() {
        let converter = FacsToFlame::default();
        let activations = FacsPresets::smile();
        let params = converter.to_flame_params(&activations);
        assert_eq!(params.expression.len(), 50);
        assert_eq!(params.pose.len(), FlameParams::NUM_JOINTS * 3);
        assert_eq!(params.translation, [0.0; 3]);
    }

    #[test]
    fn test_validate_aus_unmapped() {
        let converter = FacsToFlame::default();
        let mut activations = HashMap::new();
        activations.insert(ActionUnit::Custom(999), FacsIntensity::new(0.5));
        // Custom(999) has no registered mapping in the default library.
        let unmapped = converter.validate_aus(&activations);
        assert!(
            unmapped.contains(&ActionUnit::Custom(999)),
            "Custom(999) should be unmapped"
        );
    }

    #[test]
    fn test_validate_aus_all_mapped() {
        let converter = FacsToFlame::default();
        let activations = FacsPresets::smile();
        let unmapped = converter.validate_aus(&activations);
        assert!(unmapped.is_empty(), "all smile AUs should be mapped");
    }

    // ---- FacsPresets ------------------------------------------------------

    #[test]
    fn test_preset_neutral_empty() {
        let neutral = FacsPresets::neutral();
        assert!(
            neutral.is_empty(),
            "neutral preset must have no activations"
        );
    }

    #[test]
    fn test_preset_smile() {
        let smile = FacsPresets::smile();
        assert!(smile.contains_key(&ActionUnit::CheekRaiser));
        assert!(smile.contains_key(&ActionUnit::LipCornerPuller));
        let cheek = smile[&ActionUnit::CheekRaiser].value();
        assert!(
            (cheek - 0.8).abs() < 1e-6,
            "smile cheek raiser intensity should be 0.8"
        );
    }

    #[test]
    fn test_preset_by_name() {
        assert!(FacsPresets::by_name("smile").is_some());
        assert!(FacsPresets::by_name("SMILE").is_some());
        assert!(FacsPresets::by_name("Surprise").is_some());
        assert!(FacsPresets::by_name("unknown_preset").is_none());
    }

    #[test]
    fn test_preset_names_count() {
        let names = FacsPresets::preset_names();
        assert_eq!(names.len(), 8);
        assert!(names.contains(&"neutral"));
        assert!(names.contains(&"smile"));
        assert!(names.contains(&"blink"));
    }

    #[test]
    fn test_preset_blink() {
        let blink = FacsPresets::blink();
        assert_eq!(blink.len(), 1);
        let v = blink[&ActionUnit::Blink].value();
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_au_expression_zero_intensity() {
        let converter = FacsToFlame::default();
        let expr = converter.single_au_expression(ActionUnit::Blink, 0.0);
        assert!(expr.iter().all(|&v| v == 0.0), "zero intensity → all zeros");
    }

    #[test]
    fn test_facs_to_flame_ignores_unmapped() {
        // AUs without a mapping should not cause errors; results unchanged.
        let lib = FacsLibrary::default_flame();
        // Do not add mapping for Custom(42).
        let converter = FacsToFlame::new(lib, 50);
        let mut activations = HashMap::new();
        activations.insert(ActionUnit::Custom(42), FacsIntensity::new(1.0));
        let expr = converter.apply_aus(&activations);
        assert!(
            expr.iter().all(|&v| v == 0.0),
            "unmapped AU should produce zero expression"
        );
    }

    #[test]
    fn test_facs_to_flame_expr_dims_respected() {
        // Use a small expr_dims to verify output is truncated.
        let lib = FacsLibrary::default_flame();
        let converter = FacsToFlame::new(lib, 10);
        let activations = FacsPresets::surprise();
        let expr = converter.apply_aus(&activations);
        assert_eq!(expr.len(), 10);
    }
}
