//! Phoneme-to-FLAME-parameter mapping and speech animation synthesis.
//!
//! Maps English phoneme symbols to FLAME jaw and expression parameters,
//! enabling speech-driven avatar animation via viseme-based blending and
//! coarticulation smoothing.
//!
//! # Calibration status
//!
//! Only `jaw_angle` is physically grounded: it drives the real FLAME jaw joint
//! (`pose[6]`).  FLAME's *expression* parameters are coefficients of a learned
//! PCA basis whose components carry no articulatory meaning, so writing lip
//! protrusion or mouth width into individual coefficients — which
//! [`PhonemeParams::from_viseme`] does — is an explicitly uncalibrated
//! placeholder.  For lip-sync grounded in the FLAME basis, fit each viseme's
//! target mesh with `blend_shape_solver::fit_expression_coefficients`, collect
//! the vectors in a [`VisemeExpressionTargets`], and install them with
//! [`PhonemeLibrary::apply_expression_targets`].
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_flame::phoneme_animation::{
//!     PhonemeLibrary, parse_phoneme_string, synthesize_phoneme_animation,
//! };
//!
//! let library = PhonemeLibrary::default_english(10);
//! let events = parse_phoneme_string("p-ae-t", 0.1, &library).unwrap();
//! let clip = synthesize_phoneme_animation(&events, &library, 30.0).unwrap();
//! println!("Clip has {} keyframes", clip.n_keyframes());
//! ```

use std::collections::HashMap;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during phoneme animation operations.
#[derive(Debug, thiserror::Error)]
pub enum PhonemeError {
    /// A phoneme symbol was not found in the library.
    #[error("Unknown phoneme: {0}")]
    UnknownPhoneme(String),
    /// The phoneme event sequence is empty.
    #[error("Empty phoneme sequence")]
    EmptySequence,
    /// A duration value is not positive.
    #[error("Duration must be positive, got {0}")]
    InvalidDuration(f32),
    /// Expression coefficient count does not match the library's.
    #[error("Expression coefficient count mismatch: expected {expected}, got {got}")]
    ExpressionMismatch { expected: usize, got: usize },
    /// An invalid parameter was supplied.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ---------------------------------------------------------------------------
// Viseme
// ---------------------------------------------------------------------------

/// Standard phoneme categories mapped to viseme classes.
///
/// Visemes are the visual counterpart of phonemes — the mouth shapes
/// that correspond to groups of phonetic sounds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Viseme {
    /// No mouth movement (silence, pauses).
    Silence,
    /// p, b, m — lips together (bilabial stop/nasal).
    Bilabial,
    /// f, v — upper teeth on lower lip (labiodental fricative).
    Labiodental,
    /// th, dh — tongue at teeth (dental fricative).
    Dental,
    /// t, d, n, l, s, z, r — tongue at alveolar ridge.
    Alveolar,
    /// sh, ch, j, y — tongue raised toward palate.
    Palatal,
    /// k, g, ng — tongue at velum (velar stop/nasal).
    Velar,
    /// h — glottal fricative.
    Glottal,
    /// a, ah — wide open jaw (low vowel).
    OpenVowel,
    /// e, eh, uh — mid jaw (mid vowel).
    MidVowel,
    /// i, ee — jaw nearly closed, lips spread (high front vowel).
    ClosedVowel,
    /// o, oo, u — jaw mid, lips rounded (back/round vowel).
    RoundedVowel,
}

impl Viseme {
    /// Get jaw opening (0.0 = closed, 1.0 = max open) for this viseme.
    #[inline]
    #[must_use]
    pub fn jaw_opening(&self) -> f32 {
        match self {
            Self::Silence => 0.0,
            Self::Bilabial => 0.05,
            Self::Labiodental | Self::ClosedVowel => 0.1,
            Self::Dental | Self::Glottal => 0.2,
            Self::Alveolar => 0.25,
            Self::Palatal | Self::Velar => 0.3,
            Self::OpenVowel => 0.8,
            Self::MidVowel => 0.45,
            Self::RoundedVowel => 0.35,
        }
    }

    /// Get lip protrusion for this viseme.
    ///
    /// Returns a value where -0.5 = retracted, 0.0 = neutral, 1.0 = protruded.
    #[inline]
    #[must_use]
    pub fn lip_protrusion(&self) -> f32 {
        match self {
            Self::Bilabial => -0.1,
            Self::Labiodental => -0.2,
            Self::Silence
            | Self::Dental
            | Self::Alveolar
            | Self::Velar
            | Self::Glottal
            | Self::OpenVowel
            | Self::MidVowel => 0.0,
            Self::Palatal => 0.1,
            Self::ClosedVowel => -0.3,
            Self::RoundedVowel => 0.7,
        }
    }

    /// Get mouth width factor (1.0 = neutral, <1.0 = rounded, >1.0 = spread).
    #[inline]
    #[must_use]
    pub fn mouth_width(&self) -> f32 {
        match self {
            Self::Bilabial => 0.9,
            Self::Labiodental => 0.95,
            Self::Silence
            | Self::Dental
            | Self::Alveolar
            | Self::Velar
            | Self::Glottal
            | Self::MidVowel => 1.0,
            Self::Palatal => 1.1,
            Self::OpenVowel => 1.05,
            Self::ClosedVowel => 1.2,
            Self::RoundedVowel => 0.8,
        }
    }

    /// Convert an English phoneme symbol (ARPAbet-like) to a [`Viseme`].
    ///
    /// Returns `None` for unrecognised symbols.
    #[must_use]
    pub fn from_phoneme(phoneme: &str) -> Option<Viseme> {
        match phoneme {
            // Silence / pause markers
            "sil" | "sp" | "" => Some(Self::Silence),
            // Bilabial stops and nasal
            "p" | "b" | "m" => Some(Self::Bilabial),
            // Labiodental fricatives
            "f" | "v" => Some(Self::Labiodental),
            // Dental fricatives
            "th" | "dh" => Some(Self::Dental),
            // Alveolar consonants
            "t" | "d" | "n" | "l" | "s" | "z" | "r" => Some(Self::Alveolar),
            // Palatal fricatives and approximants
            "sh" | "zh" | "ch" | "jh" | "y" => Some(Self::Palatal),
            // Velar stops and nasal
            "k" | "g" | "ng" => Some(Self::Velar),
            // Glottal fricative
            "h" => Some(Self::Glottal),
            // Low/open vowels
            "aa" | "ah" | "ae" => Some(Self::OpenVowel),
            // Mid vowels
            "eh" | "ey" | "uh" | "er" => Some(Self::MidVowel),
            // High front vowels
            "ih" | "iy" => Some(Self::ClosedVowel),
            // Back/round vowels
            "ow" | "uw" | "ao" | "oy" => Some(Self::RoundedVowel),
            _ => None,
        }
    }

    /// Get all phoneme symbols that map to this viseme.
    #[must_use]
    pub fn phoneme_symbols(&self) -> &[&str] {
        match self {
            Self::Silence => &["sil", "sp", ""],
            Self::Bilabial => &["p", "b", "m"],
            Self::Labiodental => &["f", "v"],
            Self::Dental => &["th", "dh"],
            Self::Alveolar => &["t", "d", "n", "l", "s", "z", "r"],
            Self::Palatal => &["sh", "zh", "ch", "jh", "y"],
            Self::Velar => &["k", "g", "ng"],
            Self::Glottal => &["h"],
            Self::OpenVowel => &["aa", "ah", "ae"],
            Self::MidVowel => &["eh", "ey", "uh", "er"],
            Self::ClosedVowel => &["ih", "iy"],
            Self::RoundedVowel => &["ow", "uw", "ao", "oy"],
        }
    }
}

// ---------------------------------------------------------------------------
// VisemeExpressionTargets
// ---------------------------------------------------------------------------

/// Per-viseme targets expressed in the FLAME expression basis.
///
/// FLAME expression parameters are coefficients of a learned PCA basis whose
/// components carry no "lip protrusion" or "mouth width" semantics, so a
/// viseme's real target has to be *fitted*: take the mesh (or measured lip
/// landmarks) of that mouth shape, solve for its coefficients with
/// `blend_shape_solver::fit_expression_coefficients`, and store the resulting
/// full vector here.  Install the set with
/// [`PhonemeLibrary::apply_expression_targets`], which replaces the placeholder
/// vectors produced by [`PhonemeParams::from_viseme`].
#[derive(Debug, Clone)]
pub struct VisemeExpressionTargets {
    targets: HashMap<Viseme, Vec<f32>>,
    n_expression_coeffs: usize,
}

impl VisemeExpressionTargets {
    /// Create an empty target set for `n_expression_coeffs`-long vectors.
    #[must_use]
    pub fn new(n_expression_coeffs: usize) -> Self {
        Self {
            targets: HashMap::new(),
            n_expression_coeffs,
        }
    }

    /// Number of expression coefficients every target carries.
    #[inline]
    #[must_use]
    pub fn n_expression_coeffs(&self) -> usize {
        self.n_expression_coeffs
    }

    /// Record the fitted coefficient vector for `viseme`.
    ///
    /// # Errors
    ///
    /// Returns [`PhonemeError::ExpressionMismatch`] if `coefficients` does not
    /// have exactly [`Self::n_expression_coeffs`] entries.
    pub fn insert(&mut self, viseme: Viseme, coefficients: Vec<f32>) -> Result<(), PhonemeError> {
        if coefficients.len() != self.n_expression_coeffs {
            return Err(PhonemeError::ExpressionMismatch {
                expected: self.n_expression_coeffs,
                got: coefficients.len(),
            });
        }
        self.targets.insert(viseme, coefficients);
        Ok(())
    }

    /// Fitted coefficients recorded for `viseme`, if any.
    #[must_use]
    pub fn get(&self, viseme: &Viseme) -> Option<&[f32]> {
        self.targets.get(viseme).map(Vec::as_slice)
    }

    /// Number of visemes with a recorded target.
    #[must_use]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Is no target recorded yet?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

// ---------------------------------------------------------------------------
// PhonemeParams
// ---------------------------------------------------------------------------

/// FLAME parameters driven by a single phoneme/viseme target.
#[derive(Debug, Clone)]
pub struct PhonemeParams {
    /// Jaw rotation around the X-axis in radians (positive = open).
    pub jaw_angle: f32,
    /// Expression blend-shape coefficients (first 10 most relevant for speech).
    pub expression: Vec<f32>,
    /// Default duration in seconds for this phoneme.
    pub duration: f32,
    /// Coarticulation blend window (seconds) before/after phoneme onset/offset.
    pub coarticulation: f32,
}

impl PhonemeParams {
    /// Create a silent (zero) parameter set with `n_expr` expression coefficients.
    #[must_use]
    pub fn silence(n_expr: usize) -> Self {
        Self {
            jaw_angle: 0.0,
            expression: vec![0.0; n_expr],
            duration: 0.1,
            coarticulation: 0.05,
        }
    }

    /// Create parameters from a [`Viseme`] scaled by `intensity` (0–1).
    ///
    /// - `jaw_angle` = `viseme.jaw_opening() * intensity * 0.3`
    /// - `expr[0]` = `viseme.lip_protrusion() * intensity * 0.5`
    /// - `expr[1]` = `(viseme.mouth_width() - 1.0) * intensity * 0.3`
    /// - `expr[2..]` = `0.0`
    ///
    /// # Calibration warning
    ///
    /// Only `jaw_angle` is calibrated — it drives the real FLAME jaw joint.
    /// The expression assignment is a **placeholder with no grounding in the
    /// FLAME basis**: components 0 and 1 are the two highest-variance modes of
    /// the training corpus and encode neither lip protrusion nor mouth width,
    /// so they produce whatever deformation those modes happen to carry; the
    /// scale factors (`0.5`, `0.3`) are likewise unmeasured.  Treat the output
    /// as jaw-driven mouth motion, not as lip-synced animation, and prefer
    /// [`PhonemeParams::from_viseme_calibrated`] /
    /// [`PhonemeLibrary::apply_expression_targets`] with fitted
    /// [`VisemeExpressionTargets`].
    #[must_use]
    pub fn from_viseme(viseme: &Viseme, n_expr: usize, intensity: f32) -> Self {
        let intensity = intensity.clamp(0.0, 1.0);
        let jaw_angle = viseme.jaw_opening() * intensity * 0.3;
        let mut expression = vec![0.0_f32; n_expr];
        if n_expr >= 1 {
            expression[0] = viseme.lip_protrusion() * intensity * 0.5;
        }
        if n_expr >= 2 {
            expression[1] = (viseme.mouth_width() - 1.0) * intensity * 0.3;
        }
        Self {
            jaw_angle,
            expression,
            duration: 0.1,
            coarticulation: 0.05,
        }
    }

    /// Create parameters from a *calibrated* viseme target scaled by `intensity`.
    ///
    /// The expression vector is the fitted point in FLAME expression space for
    /// this viseme (see [`VisemeExpressionTargets`]) scaled by `intensity`, so
    /// no articulatory quantity is written into an arbitrary basis component.
    /// `jaw_angle` keeps the same calibrated formula as
    /// [`PhonemeParams::from_viseme`].
    ///
    /// # Errors
    ///
    /// Returns [`PhonemeError::InvalidParam`] if `targets` has no entry for
    /// `viseme`.
    pub fn from_viseme_calibrated(
        viseme: &Viseme,
        targets: &VisemeExpressionTargets,
        intensity: f32,
    ) -> Result<Self, PhonemeError> {
        let intensity = intensity.clamp(0.0, 1.0);
        let coefficients = targets.get(viseme).ok_or_else(|| {
            PhonemeError::InvalidParam(format!(
                "no calibrated expression target for viseme {viseme:?}"
            ))
        })?;
        Ok(Self {
            jaw_angle: viseme.jaw_opening() * intensity * 0.3,
            expression: coefficients.iter().map(|c| c * intensity).collect(),
            duration: 0.1,
            coarticulation: 0.05,
        })
    }
}

// ---------------------------------------------------------------------------
// PhonemeEvent
// ---------------------------------------------------------------------------

/// A single phoneme event within a timed sequence.
#[derive(Debug, Clone)]
pub struct PhonemeEvent {
    /// Start time of this event in seconds.
    pub start_time: f32,
    /// Duration of this event in seconds.
    pub duration: f32,
    /// Phoneme symbol string (e.g. `"p"`, `"ae"`, `"sil"`).
    pub phoneme: String,
    /// The viseme class this phoneme belongs to.
    pub viseme: Viseme,
    /// Articulation intensity in [0.0, 1.0].
    pub intensity: f32,
}

// ---------------------------------------------------------------------------
// PhonemeKeyframe
// ---------------------------------------------------------------------------

/// A single keyframe in a phoneme animation, storing jaw and expression state.
#[derive(Debug, Clone)]
pub struct PhonemeKeyframe {
    /// Time of this keyframe in seconds.
    pub time: f32,
    /// Jaw rotation in radians at this keyframe.
    pub jaw_angle: f32,
    /// Expression blend-shape coefficients at this keyframe.
    pub expression: Vec<f32>,
}

// ---------------------------------------------------------------------------
// PhonemeClip
// ---------------------------------------------------------------------------

/// A complete phoneme animation clip containing events and keyframes.
#[derive(Debug, Clone)]
pub struct PhonemeClip {
    /// The source phoneme events.
    pub events: Vec<PhonemeEvent>,
    /// Sampled keyframes at `sample_rate` frames per second.
    pub keyframes: Vec<PhonemeKeyframe>,
    /// Total duration in seconds.
    pub total_duration: f32,
    /// Number of expression coefficients per keyframe.
    pub n_expression_coeffs: usize,
    /// Keyframes per second.
    pub sample_rate: f32,
}

impl PhonemeClip {
    /// Sample the clip at time `t` by linear interpolation between adjacent keyframes.
    ///
    /// Clamps `t` to `[0, total_duration]`.
    #[must_use]
    pub fn sample_at(&self, time: f32) -> PhonemeKeyframe {
        if self.keyframes.is_empty() {
            return PhonemeKeyframe {
                time,
                jaw_angle: 0.0,
                expression: vec![0.0; self.n_expression_coeffs],
            };
        }
        let t = time.clamp(0.0, self.total_duration);

        // Binary-search for the right interval
        let idx = self
            .keyframes
            .partition_point(|kf| kf.time <= t)
            .saturating_sub(1);

        let idx = idx.min(self.keyframes.len() - 1);

        if idx + 1 >= self.keyframes.len() {
            let last = &self.keyframes[idx];
            return PhonemeKeyframe {
                time: t,
                jaw_angle: last.jaw_angle,
                expression: last.expression.clone(),
            };
        }

        let a = &self.keyframes[idx];
        let b = &self.keyframes[idx + 1];
        let span = b.time - a.time;
        let alpha = if span.abs() < 1e-9 {
            0.0
        } else {
            ((t - a.time) / span).clamp(0.0, 1.0)
        };

        let jaw_angle = a.jaw_angle + alpha * (b.jaw_angle - a.jaw_angle);
        let expression = a
            .expression
            .iter()
            .zip(b.expression.iter())
            .map(|(av, bv)| av + alpha * (bv - av))
            .collect();

        PhonemeKeyframe {
            time: t,
            jaw_angle,
            expression,
        }
    }

    /// Total duration of the clip.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.total_duration
    }

    /// Number of keyframes in the clip.
    #[inline]
    #[must_use]
    pub fn n_keyframes(&self) -> usize {
        self.keyframes.len()
    }
}

// ---------------------------------------------------------------------------
// PhonemeLibrary
// ---------------------------------------------------------------------------

/// Library mapping phoneme strings to pre-defined [`PhonemeParams`].
pub struct PhonemeLibrary {
    params: HashMap<String, PhonemeParams>,
    /// Number of expression coefficients used across all entries.
    pub n_expression_coeffs: usize,
}

impl PhonemeLibrary {
    /// Create an empty library with `n_expression_coeffs` expression coefficients.
    #[must_use]
    pub fn new(n_expression_coeffs: usize) -> Self {
        Self {
            params: HashMap::new(),
            n_expression_coeffs,
        }
    }

    /// Create a library pre-populated with default English phoneme mappings.
    ///
    /// Intensity is set to `1.0` (full articulation) for all consonants and
    /// vowels, and `0.0` for silence markers.
    ///
    /// The jaw angles are calibrated; the expression vectors come from
    /// [`PhonemeParams::from_viseme`] and are an uncalibrated placeholder.  Call
    /// [`Self::apply_expression_targets`] with fitted
    /// [`VisemeExpressionTargets`] to replace them.
    #[must_use]
    pub fn default_english(n_expression_coeffs: usize) -> Self {
        tracing::warn!(
            "PhonemeLibrary::default_english: expression vectors are an uncalibrated \
             placeholder with no grounding in the FLAME expression basis (only jaw \
             angles are calibrated); use apply_expression_targets for real lip-sync"
        );
        let mut lib = Self::new(n_expression_coeffs);

        // Silence
        for sym in ["sil", "sp", ""] {
            let p = PhonemeParams::from_viseme(&Viseme::Silence, n_expression_coeffs, 0.0);
            lib.params.insert(sym.to_string(), p);
        }

        // All non-silence viseme classes with intensity 1.0
        let viseme_phoneme_pairs: &[(&Viseme, &[&str])] = &[
            (&Viseme::Bilabial, &["p", "b", "m"]),
            (&Viseme::Labiodental, &["f", "v"]),
            (&Viseme::Dental, &["th", "dh"]),
            (&Viseme::Alveolar, &["t", "d", "n", "l", "s", "z", "r"]),
            (&Viseme::Palatal, &["sh", "zh", "ch", "jh", "y"]),
            (&Viseme::Velar, &["k", "g", "ng"]),
            (&Viseme::Glottal, &["h"]),
            (&Viseme::OpenVowel, &["aa", "ah", "ae"]),
            (&Viseme::MidVowel, &["eh", "ey", "uh", "er"]),
            (&Viseme::ClosedVowel, &["ih", "iy"]),
            (&Viseme::RoundedVowel, &["ow", "uw", "ao", "oy"]),
        ];

        for (viseme, symbols) in viseme_phoneme_pairs {
            let p = PhonemeParams::from_viseme(viseme, n_expression_coeffs, 1.0);
            for sym in *symbols {
                lib.params.insert((*sym).to_string(), p.clone());
            }
        }

        lib
    }

    /// Look up parameters for a phoneme symbol.
    #[must_use]
    pub fn get(&self, phoneme: &str) -> Option<&PhonemeParams> {
        self.params.get(phoneme)
    }

    /// Register a custom phoneme entry.
    ///
    /// Returns [`PhonemeError::ExpressionMismatch`] if the expression length
    /// in `params` does not match `self.n_expression_coeffs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn register(&mut self, phoneme: String, params: PhonemeParams) -> Result<(), PhonemeError> {
        if params.expression.len() != self.n_expression_coeffs {
            return Err(PhonemeError::ExpressionMismatch {
                expected: self.n_expression_coeffs,
                got: params.expression.len(),
            });
        }
        self.params.insert(phoneme, params);
        Ok(())
    }

    /// Number of phoneme entries in the library.
    #[must_use]
    pub fn n_phonemes(&self) -> usize {
        self.params.len()
    }

    /// Replace every entry's expression vector with the calibrated target for
    /// its viseme, keeping the calibrated jaw angles and timings.
    ///
    /// This is the supported way to turn a placeholder library (see
    /// [`Self::default_english`]) into one whose expression coefficients are
    /// real points in the FLAME expression basis.
    ///
    /// # Errors
    ///
    /// - [`PhonemeError::ExpressionMismatch`] — `targets` has a different
    ///   coefficient count than this library.
    /// - [`PhonemeError::UnknownPhoneme`] — an entry's symbol has no viseme.
    /// - [`PhonemeError::InvalidParam`] — a required viseme has no target.
    pub fn apply_expression_targets(
        &mut self,
        targets: &VisemeExpressionTargets,
    ) -> Result<(), PhonemeError> {
        if targets.n_expression_coeffs() != self.n_expression_coeffs {
            return Err(PhonemeError::ExpressionMismatch {
                expected: self.n_expression_coeffs,
                got: targets.n_expression_coeffs(),
            });
        }
        for (phoneme, params) in &mut self.params {
            let viseme = Viseme::from_phoneme(phoneme)
                .ok_or_else(|| PhonemeError::UnknownPhoneme(phoneme.clone()))?;
            let coefficients = targets.get(&viseme).ok_or_else(|| {
                PhonemeError::InvalidParam(format!(
                    "no calibrated expression target for viseme {viseme:?}"
                ))
            })?;
            params.expression.clear();
            params.expression.extend_from_slice(coefficients);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// synthesize_phoneme_animation
// ---------------------------------------------------------------------------

/// Convert a phoneme event sequence into a keyframe animation clip.
///
/// For each frame time the active event(s) are blended, and coarticulation
/// windows cause adjacent phonemes to bleed into one another.
///
/// # Errors
///
/// - [`PhonemeError::EmptySequence`] — `events` is empty.
/// - [`PhonemeError::UnknownPhoneme`] — any event's phoneme is not in `library`.
/// - [`PhonemeError::InvalidDuration`] — `sample_rate` ≤ 0.
pub fn synthesize_phoneme_animation(
    events: &[PhonemeEvent],
    library: &PhonemeLibrary,
    sample_rate: f32,
) -> Result<PhonemeClip, PhonemeError> {
    if events.is_empty() {
        return Err(PhonemeError::EmptySequence);
    }
    if sample_rate <= 0.0 {
        return Err(PhonemeError::InvalidDuration(sample_rate));
    }

    // Validate all phonemes are in library
    for ev in events {
        if library.get(&ev.phoneme).is_none() {
            return Err(PhonemeError::UnknownPhoneme(ev.phoneme.clone()));
        }
    }

    let n_expr = library.n_expression_coeffs;
    let total_duration = events
        .iter()
        .map(|e| e.start_time + e.duration)
        .fold(0.0_f32, f32::max);

    let n_frames = ((total_duration * sample_rate).ceil() as usize).max(1);
    let mut keyframes = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let t = frame_idx as f32 / sample_rate;
        let (jaw_angle, expression) = sample_events_at(t, events, library, n_expr);
        keyframes.push(PhonemeKeyframe {
            time: t,
            jaw_angle,
            expression,
        });
    }

    Ok(PhonemeClip {
        events: events.to_vec(),
        keyframes,
        total_duration,
        n_expression_coeffs: n_expr,
        sample_rate,
    })
}

/// Evaluate FLAME parameters at time `t` over `events`, blending coarticulation windows.
fn sample_events_at(
    t: f32,
    events: &[PhonemeEvent],
    library: &PhonemeLibrary,
    n_expr: usize,
) -> (f32, Vec<f32>) {
    let mut total_weight = 0.0_f32;
    let mut jaw_acc = 0.0_f32;
    let mut expr_acc = vec![0.0_f32; n_expr];

    for ev in events {
        let end_time = ev.start_time + ev.duration;
        // Use coarticulation from library params, falling back to a default
        let coart = library
            .get(&ev.phoneme)
            .map_or(0.05, |p| p.coarticulation)
            .max(1e-6);

        // Compute influence weight: 1.0 inside event, tapering off in coarticulation windows
        let weight = coarticulation_weight(t, ev.start_time, end_time, coart);
        if weight <= 0.0 {
            continue;
        }

        // Retrieve pre-built params from library (already validated)
        let Some(params) = library.get(&ev.phoneme) else {
            continue;
        };

        // Scale by event intensity
        let scaled_jaw = params.jaw_angle * ev.intensity;
        jaw_acc += weight * scaled_jaw;

        for (acc, val) in expr_acc.iter_mut().zip(params.expression.iter()) {
            *acc += weight * val * ev.intensity;
        }
        total_weight += weight;
    }

    if total_weight < 1e-9 {
        return (0.0, vec![0.0; n_expr]);
    }

    let inv = 1.0 / total_weight;
    let jaw_angle = jaw_acc * inv;
    let expression = expr_acc.iter().map(|v| v * inv).collect();
    (jaw_angle, expression)
}

/// Smooth "tent" weight: 1.0 inside [start, end], linearly falling over `coart` seconds.
#[inline]
fn coarticulation_weight(t: f32, start: f32, end: f32, coart: f32) -> f32 {
    if t < start - coart || t > end + coart {
        return 0.0;
    }
    if t >= start && t <= end {
        return 1.0;
    }
    if t < start {
        return (t - (start - coart)) / coart;
    }
    // t > end
    (end + coart - t) / coart
}

// ---------------------------------------------------------------------------
// parse_phoneme_string
// ---------------------------------------------------------------------------

/// Parse a dash-separated phoneme string such as `"p-ae-t"` into events.
///
/// Each phoneme gets a fixed `phoneme_duration` and is placed sequentially.
///
/// # Errors
///
/// - [`PhonemeError::EmptySequence`] — the string is empty.
/// - [`PhonemeError::InvalidDuration`] — `phoneme_duration` ≤ 0.
/// - [`PhonemeError::UnknownPhoneme`] — a symbol is not in the library.
pub fn parse_phoneme_string(
    phoneme_str: &str,
    phoneme_duration: f32,
    library: &PhonemeLibrary,
) -> Result<Vec<PhonemeEvent>, PhonemeError> {
    if phoneme_str.is_empty() {
        return Err(PhonemeError::EmptySequence);
    }
    if phoneme_duration <= 0.0 {
        return Err(PhonemeError::InvalidDuration(phoneme_duration));
    }

    let symbols: Vec<&str> = phoneme_str.split('-').collect();
    if symbols.is_empty() {
        return Err(PhonemeError::EmptySequence);
    }

    let mut events = Vec::with_capacity(symbols.len());
    let mut start_time = 0.0_f32;

    for sym in &symbols {
        let viseme = Viseme::from_phoneme(sym)
            .ok_or_else(|| PhonemeError::UnknownPhoneme((*sym).to_string()))?;

        // Validate against library too
        if library.get(sym).is_none() {
            return Err(PhonemeError::UnknownPhoneme((*sym).to_string()));
        }

        let intensity = if viseme == Viseme::Silence { 0.0 } else { 1.0 };

        events.push(PhonemeEvent {
            start_time,
            duration: phoneme_duration,
            phoneme: (*sym).to_string(),
            viseme,
            intensity,
        });
        start_time += phoneme_duration;
    }

    Ok(events)
}

// ---------------------------------------------------------------------------
// apply_coarticulation
// ---------------------------------------------------------------------------

/// Apply coarticulation smoothing: blend each keyframe with a Gaussian window
/// of nearby frames, modelling the temporal smearing of articulatory targets.
///
/// `coarticulation_window` is in seconds; `sample_rate` converts to frames.
#[must_use]
pub fn apply_coarticulation(
    keyframes: &[PhonemeKeyframe],
    coarticulation_window: f32,
    sample_rate: f32,
) -> Vec<PhonemeKeyframe> {
    if keyframes.is_empty() {
        return Vec::new();
    }

    let sigma = (coarticulation_window * sample_rate).max(0.5);
    smooth_with_gaussian(keyframes, sigma)
}

// ---------------------------------------------------------------------------
// smooth_phoneme_keyframes
// ---------------------------------------------------------------------------

/// Smooth keyframe sequence with a Gaussian kernel of `sigma_frames` frame-width.
///
/// When `sigma_frames` ≤ 0.0 the input is returned unchanged (modulo clone).
#[must_use]
pub fn smooth_phoneme_keyframes(
    keyframes: &[PhonemeKeyframe],
    sigma_frames: f32,
) -> Vec<PhonemeKeyframe> {
    if keyframes.is_empty() || sigma_frames <= 0.0 {
        return keyframes.to_vec();
    }
    smooth_with_gaussian(keyframes, sigma_frames)
}

/// Shared Gaussian smoothing kernel applied over a keyframe slice.
fn smooth_with_gaussian(keyframes: &[PhonemeKeyframe], sigma: f32) -> Vec<PhonemeKeyframe> {
    let n = keyframes.len();
    if n == 0 {
        return Vec::new();
    }
    let n_expr = keyframes[0].expression.len();

    // Half-window: 3 sigma, minimum 1
    let half = ((3.0 * sigma).ceil() as usize).max(1);

    // Pre-compute Gaussian weights
    let build_weights = |half: usize, sigma: f32| -> Vec<f32> {
        let len = 2 * half + 1;
        let mut w = Vec::with_capacity(len);
        for di in 0..len {
            let d = di as f32 - half as f32;
            w.push((-0.5 * (d / sigma) * (d / sigma)).exp());
        }
        w
    };
    let weights = build_weights(half, sigma);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut jaw_acc = 0.0_f32;
        let mut expr_acc = vec![0.0_f32; n_expr];
        let mut w_sum = 0.0_f32;

        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);

        for j in lo..hi {
            let wi = weights[j + half - i];
            jaw_acc += wi * keyframes[j].jaw_angle;
            for (acc, val) in expr_acc.iter_mut().zip(keyframes[j].expression.iter()) {
                *acc += wi * val;
            }
            w_sum += wi;
        }

        let inv = if w_sum > 1e-12 { 1.0 / w_sum } else { 1.0 };
        out.push(PhonemeKeyframe {
            time: keyframes[i].time,
            jaw_angle: jaw_acc * inv,
            expression: expr_acc.iter().map(|v| v * inv).collect(),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// generate_breath_animation
// ---------------------------------------------------------------------------

/// Generate subtle breathing (idle) animation without any phoneme input.
///
/// Uses a deterministic sinusoidal model: the jaw completes exactly one
/// open/close cycle per breath, i.e. `breath_rate` cycles per minute, and a
/// secondary component adds micro-expression variation on expression
/// coefficients.
///
/// The jaw curve is `sin²(φ/2) = (1 - cos φ) / 2` with `φ = 2π · f · t` and
/// `f = breath_rate / 60`.  Halving the phase inside the square keeps the jaw
/// non-negative (it only ever opens) *without* halving the period — squaring
/// the full-phase sine would double the visible breathing rate.
///
/// `expression[0]` follows the raw `sin φ` fundamental and `expression[1]`
/// carries the genuine second harmonic `sin 2φ`.
///
/// `amplitude` scales the overall magnitude (0.0–1.0 recommended).
#[must_use]
pub fn generate_breath_animation(
    duration: f32,
    breath_rate: f32,
    n_expr: usize,
    amplitude: f32,
) -> Vec<PhonemeKeyframe> {
    if duration <= 0.0 || n_expr == 0 {
        return Vec::new();
    }

    // 30 fps default for breath animation
    let sample_rate = 30.0_f32;
    let n_frames = ((duration * sample_rate).ceil() as usize).max(1);
    let freq = breath_rate / 60.0; // breaths per second

    let mut keyframes = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let t = frame_idx as f32 / sample_rate;
        let phase = 2.0 * PI * freq * t;
        let breath = phase.sin();
        // Half-phase squared sine: non-negative (the jaw only opens) *and*
        // period-preserving, so the jaw cycles at `breath_rate` per minute.
        let half_breath = (0.5 * phase).sin();
        let jaw_angle = amplitude * half_breath * half_breath * 0.05;

        let mut expression = vec![0.0_f32; n_expr];
        // Subtle chest/throat expansion on expr[0]
        if n_expr >= 1 {
            expression[0] = amplitude * breath * 0.01;
        }
        // Second harmonic on expr[1] for realism (2x the jaw's fundamental)
        if n_expr >= 2 {
            expression[1] = amplitude * (2.0 * phase).sin() * 0.005;
        }

        keyframes.push(PhonemeKeyframe {
            time: t,
            jaw_angle,
            expression,
        });
    }

    keyframes
}

// ---------------------------------------------------------------------------
// blend_phoneme_with_base
// ---------------------------------------------------------------------------

/// Blend a phoneme animation sequence with a static base expression.
///
/// At `blend_weight = 1.0` the output equals `phoneme_frames` unchanged.
/// At `blend_weight = 0.0` the output equals the base parameters.
///
/// # Errors
///
/// - [`PhonemeError::ExpressionMismatch`] — `base_expression` length mismatches
///   the first keyframe's expression length.
pub fn blend_phoneme_with_base(
    phoneme_frames: &[PhonemeKeyframe],
    base_expression: &[f32],
    base_jaw_angle: f32,
    blend_weight: f32,
) -> Result<Vec<PhonemeKeyframe>, PhonemeError> {
    if phoneme_frames.is_empty() {
        return Ok(Vec::new());
    }

    let n_expr = phoneme_frames[0].expression.len();
    if base_expression.len() != n_expr {
        return Err(PhonemeError::ExpressionMismatch {
            expected: n_expr,
            got: base_expression.len(),
        });
    }

    let w = blend_weight.clamp(0.0, 1.0);
    let base_w = 1.0 - w;

    let out = phoneme_frames
        .iter()
        .map(|kf| {
            let jaw_angle = w * kf.jaw_angle + base_w * base_jaw_angle;
            let expression = kf
                .expression
                .iter()
                .zip(base_expression.iter())
                .map(|(ph, base)| w * ph + base_w * base)
                .collect();
            PhonemeKeyframe {
                time: kf.time,
                jaw_angle,
                expression,
            }
        })
        .collect();

    Ok(out)
}

// ---------------------------------------------------------------------------
// extract_jaw_sequence / extract_expression_sequence
// ---------------------------------------------------------------------------

/// Extract per-frame jaw angles from a keyframe sequence.
#[must_use]
pub fn extract_jaw_sequence(keyframes: &[PhonemeKeyframe]) -> Vec<f32> {
    keyframes.iter().map(|kf| kf.jaw_angle).collect()
}

/// Extract per-frame expression coefficient vectors from a keyframe sequence.
#[must_use]
pub fn extract_expression_sequence(keyframes: &[PhonemeKeyframe]) -> Vec<Vec<f32>> {
    keyframes.iter().map(|kf| kf.expression.clone()).collect()
}

// ---------------------------------------------------------------------------
// PhonemeStats + phoneme_clip_stats
// ---------------------------------------------------------------------------

/// Summary statistics for a [`PhonemeClip`].
#[derive(Debug, Clone)]
pub struct PhonemeStats {
    /// Number of phoneme events.
    pub n_events: usize,
    /// Number of keyframes.
    pub n_keyframes: usize,
    /// Total clip duration in seconds.
    pub total_duration: f32,
    /// Mean jaw angle across all keyframes.
    pub mean_jaw_angle: f32,
    /// Maximum jaw angle across all keyframes.
    pub max_jaw_angle: f32,
    /// Number of unique viseme classes present.
    pub n_unique_visemes: usize,
    /// Fraction of events that are non-silent.
    pub speaking_fraction: f32,
}

/// Compute summary statistics for a phoneme clip.
pub fn phoneme_clip_stats(clip: &PhonemeClip) -> PhonemeStats {
    let n_events = clip.events.len();
    let n_keyframes = clip.keyframes.len();
    let total_duration = clip.total_duration;

    let (mean_jaw_angle, max_jaw_angle) = if clip.keyframes.is_empty() {
        (0.0, 0.0)
    } else {
        let sum: f32 = clip.keyframes.iter().map(|kf| kf.jaw_angle).sum();
        let mean = sum / clip.keyframes.len() as f32;
        let max = clip
            .keyframes
            .iter()
            .map(|kf| kf.jaw_angle)
            .fold(0.0_f32, f32::max);
        (mean, max)
    };

    let mut viseme_set: HashMap<String, bool> = HashMap::new();
    for ev in &clip.events {
        viseme_set.insert(format!("{:?}", ev.viseme), true);
    }
    let n_unique_visemes = viseme_set.len();

    let n_speaking = clip
        .events
        .iter()
        .filter(|ev| ev.viseme != Viseme::Silence)
        .count();
    let speaking_fraction = if n_events == 0 {
        0.0
    } else {
        n_speaking as f32 / n_events as f32
    };

    PhonemeStats {
        n_events,
        n_keyframes,
        total_duration,
        mean_jaw_angle,
        max_jaw_angle,
        n_unique_visemes,
        speaking_fraction,
    }
}

/// Format phoneme statistics as a human-readable string.
#[must_use]
pub fn format_phoneme_stats(stats: &PhonemeStats) -> String {
    format!(
        "PhonemeStats {{ events: {}, keyframes: {}, duration: {:.3}s, \
         mean_jaw: {:.4}, max_jaw: {:.4}, unique_visemes: {}, speaking: {:.1}% }}",
        stats.n_events,
        stats.n_keyframes,
        stats.total_duration,
        stats.mean_jaw_angle,
        stats.max_jaw_angle,
        stats.n_unique_visemes,
        stats.speaking_fraction * 100.0,
    )
}

/// Format a phoneme clip summary as a human-readable string.
#[must_use]
pub fn format_phoneme_clip(clip: &PhonemeClip) -> String {
    format!(
        "PhonemeClip {{ events: {}, keyframes: {}, duration: {:.3}s, \
         sample_rate: {:.1}fps, n_expr: {} }}",
        clip.events.len(),
        clip.keyframes.len(),
        clip.total_duration,
        clip.sample_rate,
        clip.n_expression_coeffs,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every viseme variant, for exhaustive property checks.
    fn all_visemes() -> [Viseme; 12] {
        [
            Viseme::Silence,
            Viseme::Bilabial,
            Viseme::Labiodental,
            Viseme::Dental,
            Viseme::Alveolar,
            Viseme::Palatal,
            Viseme::Velar,
            Viseme::Glottal,
            Viseme::OpenVowel,
            Viseme::MidVowel,
            Viseme::ClosedVowel,
            Viseme::RoundedVowel,
        ]
    }

    // --- Viseme::jaw_opening ---

    #[test]
    fn test_jaw_opening_silence_is_zero() {
        assert_eq!(Viseme::Silence.jaw_opening(), 0.0);
    }

    #[test]
    fn test_jaw_opening_open_vowel_is_highest() {
        let max = all_visemes()
            .iter()
            .map(super::Viseme::jaw_opening)
            .fold(0.0_f32, f32::max);
        assert_eq!(max, Viseme::OpenVowel.jaw_opening());
    }

    #[test]
    fn test_jaw_opening_all_variants_in_range() {
        for v in &all_visemes() {
            let o = v.jaw_opening();
            assert!((0.0..=1.0).contains(&o), "{v:?}: jaw_opening={o}");
        }
    }

    #[test]
    fn test_jaw_opening_bilabial_small() {
        assert!(Viseme::Bilabial.jaw_opening() < 0.1);
    }

    #[test]
    fn test_jaw_opening_open_vowel_value() {
        assert!((Viseme::OpenVowel.jaw_opening() - 0.8).abs() < 1e-6);
    }

    // --- Viseme::lip_protrusion ---

    #[test]
    fn test_lip_protrusion_all_finite() {
        for v in &all_visemes() {
            assert!(
                v.lip_protrusion().is_finite(),
                "{v:?}: lip_protrusion is not finite"
            );
        }
    }

    #[test]
    fn test_lip_protrusion_rounded_vowel_positive() {
        assert!(Viseme::RoundedVowel.lip_protrusion() > 0.0);
    }

    #[test]
    fn test_lip_protrusion_closed_vowel_negative() {
        assert!(Viseme::ClosedVowel.lip_protrusion() < 0.0);
    }

    // --- Viseme::mouth_width ---

    #[test]
    fn test_mouth_width_all_positive() {
        for v in &all_visemes() {
            assert!(
                v.mouth_width() > 0.0,
                "{:?}: mouth_width={} is not positive",
                v,
                v.mouth_width()
            );
        }
    }

    #[test]
    fn test_mouth_width_closed_vowel_wide() {
        // Spread lips for ClosedVowel (ee)
        assert!(Viseme::ClosedVowel.mouth_width() > 1.0);
    }

    #[test]
    fn test_mouth_width_rounded_vowel_narrow() {
        // Rounded lips for RoundedVowel (oo)
        assert!(Viseme::RoundedVowel.mouth_width() < 1.0);
    }

    // --- Viseme::from_phoneme ---

    #[test]
    fn test_from_phoneme_p_is_bilabial() {
        assert_eq!(Viseme::from_phoneme("p"), Some(Viseme::Bilabial));
    }

    #[test]
    fn test_from_phoneme_b_is_bilabial() {
        assert_eq!(Viseme::from_phoneme("b"), Some(Viseme::Bilabial));
    }

    #[test]
    fn test_from_phoneme_m_is_bilabial() {
        assert_eq!(Viseme::from_phoneme("m"), Some(Viseme::Bilabial));
    }

    #[test]
    fn test_from_phoneme_f_labiodental() {
        assert_eq!(Viseme::from_phoneme("f"), Some(Viseme::Labiodental));
    }

    #[test]
    fn test_from_phoneme_th_dental() {
        assert_eq!(Viseme::from_phoneme("th"), Some(Viseme::Dental));
    }

    #[test]
    fn test_from_phoneme_s_alveolar() {
        assert_eq!(Viseme::from_phoneme("s"), Some(Viseme::Alveolar));
    }

    #[test]
    fn test_from_phoneme_sh_palatal() {
        assert_eq!(Viseme::from_phoneme("sh"), Some(Viseme::Palatal));
    }

    #[test]
    fn test_from_phoneme_k_velar() {
        assert_eq!(Viseme::from_phoneme("k"), Some(Viseme::Velar));
    }

    #[test]
    fn test_from_phoneme_h_glottal() {
        assert_eq!(Viseme::from_phoneme("h"), Some(Viseme::Glottal));
    }

    #[test]
    fn test_from_phoneme_aa_open_vowel() {
        assert_eq!(Viseme::from_phoneme("aa"), Some(Viseme::OpenVowel));
    }

    #[test]
    fn test_from_phoneme_eh_mid_vowel() {
        assert_eq!(Viseme::from_phoneme("eh"), Some(Viseme::MidVowel));
    }

    #[test]
    fn test_from_phoneme_iy_closed_vowel() {
        assert_eq!(Viseme::from_phoneme("iy"), Some(Viseme::ClosedVowel));
    }

    #[test]
    fn test_from_phoneme_ow_rounded_vowel() {
        assert_eq!(Viseme::from_phoneme("ow"), Some(Viseme::RoundedVowel));
    }

    #[test]
    fn test_from_phoneme_sil_silence() {
        assert_eq!(Viseme::from_phoneme("sil"), Some(Viseme::Silence));
    }

    #[test]
    fn test_from_phoneme_empty_silence() {
        assert_eq!(Viseme::from_phoneme(""), Some(Viseme::Silence));
    }

    #[test]
    fn test_from_phoneme_unknown_returns_none() {
        assert_eq!(Viseme::from_phoneme("xyz_unknown"), None);
    }

    // --- PhonemeParams::silence ---

    #[test]
    fn test_silence_params_zero_jaw() {
        let p = PhonemeParams::silence(10);
        assert_eq!(p.jaw_angle, 0.0);
    }

    #[test]
    fn test_silence_params_zero_expression() {
        let p = PhonemeParams::silence(10);
        assert!(p.expression.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_silence_params_correct_length() {
        let n = 15;
        let p = PhonemeParams::silence(n);
        assert_eq!(p.expression.len(), n);
    }

    // --- PhonemeParams::from_viseme ---

    #[test]
    fn test_from_viseme_silence_zero_jaw() {
        let p = PhonemeParams::from_viseme(&Viseme::Silence, 10, 1.0);
        assert_eq!(p.jaw_angle, 0.0);
    }

    #[test]
    fn test_from_viseme_open_vowel_nonzero_jaw() {
        let p = PhonemeParams::from_viseme(&Viseme::OpenVowel, 10, 1.0);
        assert!(p.jaw_angle > 0.0);
    }

    #[test]
    fn test_from_viseme_intensity_zero_means_silence() {
        let p = PhonemeParams::from_viseme(&Viseme::OpenVowel, 10, 0.0);
        assert_eq!(p.jaw_angle, 0.0);
        assert!(p.expression.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_from_viseme_expression_length() {
        let p = PhonemeParams::from_viseme(&Viseme::MidVowel, 8, 1.0);
        assert_eq!(p.expression.len(), 8);
    }

    // --- VisemeExpressionTargets (calibrated expression mapping) ---

    #[test]
    fn test_viseme_targets_reject_wrong_length() {
        let mut targets = VisemeExpressionTargets::new(4);
        assert!(targets.is_empty());
        assert!(matches!(
            targets.insert(Viseme::Silence, vec![0.0; 3]),
            Err(PhonemeError::ExpressionMismatch {
                expected: 4,
                got: 3
            })
        ));
        let mut lib = PhonemeLibrary::default_english(6);
        assert!(matches!(
            lib.apply_expression_targets(&targets),
            Err(PhonemeError::ExpressionMismatch { .. })
        ));
    }

    #[test]
    fn test_from_viseme_calibrated_uses_fitted_vector() {
        let mut targets = VisemeExpressionTargets::new(3);
        targets
            .insert(Viseme::OpenVowel, vec![0.4, -0.2, 0.1])
            .expect("insert ok");
        let p = PhonemeParams::from_viseme_calibrated(&Viseme::OpenVowel, &targets, 0.5)
            .expect("calibrated params");
        assert_eq!(p.expression.len(), 3);
        assert!((p.expression[0] - 0.2).abs() < 1e-6, "{:?}", p.expression);
        assert!((p.expression[1] + 0.1).abs() < 1e-6, "{:?}", p.expression);
        assert!(p.jaw_angle > 0.0);
        // A viseme without a fitted target is reported, never silently faked.
        assert!(matches!(
            PhonemeParams::from_viseme_calibrated(&Viseme::Bilabial, &targets, 1.0),
            Err(PhonemeError::InvalidParam(_))
        ));
    }

    #[test]
    fn test_apply_expression_targets_replaces_placeholder() {
        // The placeholder mapping writes lip/mouth quantities into expression
        // components 0 and 1, which carry no such meaning in the FLAME PCA
        // basis.  Fitted targets replace the whole vector instead.
        let n = 6;
        let mut targets = VisemeExpressionTargets::new(n);
        for (i, viseme) in all_visemes().into_iter().enumerate() {
            targets
                .insert(viseme, vec![i as f32 * 0.1; n])
                .expect("insert ok");
        }
        assert_eq!(targets.len(), 12);

        let mut lib = PhonemeLibrary::default_english(n);
        let placeholder = lib.get("ow").expect("ow present").expression.clone();
        lib.apply_expression_targets(&targets).expect("apply ok");

        let expected = targets
            .get(&Viseme::RoundedVowel)
            .expect("rounded vowel target")
            .to_vec();
        let updated = lib.get("ow").expect("ow present");
        assert_eq!(updated.expression, expected);
        assert_ne!(updated.expression, placeholder);
        // Jaw angles stay calibrated.
        assert!(updated.jaw_angle > 0.0);
    }

    // --- PhonemeLibrary ---

    #[test]
    fn test_library_default_english_has_entries() {
        let lib = PhonemeLibrary::default_english(10);
        assert!(lib.n_phonemes() > 20);
    }

    #[test]
    fn test_library_get_p_found() {
        let lib = PhonemeLibrary::default_english(10);
        assert!(lib.get("p").is_some());
    }

    #[test]
    fn test_library_get_ae_found() {
        let lib = PhonemeLibrary::default_english(10);
        assert!(lib.get("ae").is_some());
    }

    #[test]
    fn test_library_get_unknown_none() {
        let lib = PhonemeLibrary::default_english(10);
        assert!(lib.get("xyz_not_real").is_none());
    }

    #[test]
    fn test_library_register_valid() {
        let mut lib = PhonemeLibrary::default_english(10);
        let p = PhonemeParams::silence(10);
        assert!(lib.register("custom_ph".to_string(), p).is_ok());
        assert!(lib.get("custom_ph").is_some());
    }

    #[test]
    fn test_library_register_mismatch_error() {
        let mut lib = PhonemeLibrary::default_english(10);
        let p = PhonemeParams::silence(5); // wrong length
        let result = lib.register("bad_ph".to_string(), p);
        assert!(matches!(
            result,
            Err(PhonemeError::ExpressionMismatch {
                expected: 10,
                got: 5
            })
        ));
    }

    // --- parse_phoneme_string ---

    #[test]
    fn test_parse_three_phoneme_string() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse ok");
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_parse_phoneme_correct_timing() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse ok");
        assert!((events[0].start_time - 0.0).abs() < 1e-6);
        assert!((events[1].start_time - 0.1).abs() < 1e-6);
        assert!((events[2].start_time - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_parse_unknown_phoneme_error() {
        let lib = PhonemeLibrary::default_english(10);
        let result = parse_phoneme_string("p-XXXXX-t", 0.1, &lib);
        assert!(matches!(result, Err(PhonemeError::UnknownPhoneme(_))));
    }

    #[test]
    fn test_parse_empty_string_error() {
        let lib = PhonemeLibrary::default_english(10);
        let result = parse_phoneme_string("", 0.1, &lib);
        assert!(matches!(result, Err(PhonemeError::EmptySequence)));
    }

    #[test]
    fn test_parse_invalid_duration_error() {
        let lib = PhonemeLibrary::default_english(10);
        let result = parse_phoneme_string("p-ae", -0.1, &lib);
        assert!(matches!(result, Err(PhonemeError::InvalidDuration(_))));
    }

    // --- synthesize_phoneme_animation ---

    #[test]
    fn test_synthesize_correct_duration() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        assert!((clip.total_duration - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_synthesize_correct_n_keyframes() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        // 0.3s at 30fps = 9 frames (ceil(0.3*30)=9)
        assert_eq!(clip.n_keyframes(), 9);
    }

    #[test]
    fn test_synthesize_empty_events_error() {
        let lib = PhonemeLibrary::default_english(10);
        let result = synthesize_phoneme_animation(&[], &lib, 30.0);
        assert!(matches!(result, Err(PhonemeError::EmptySequence)));
    }

    #[test]
    fn test_synthesize_invalid_sample_rate_error() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae", 0.1, &lib).expect("parse");
        let result = synthesize_phoneme_animation(&events, &lib, 0.0);
        assert!(matches!(result, Err(PhonemeError::InvalidDuration(_))));
    }

    // --- PhonemeClip::sample_at ---

    #[test]
    fn test_sample_at_t0_returns_first_frame_values() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let sampled = clip.sample_at(0.0);
        let first = &clip.keyframes[0];
        assert!((sampled.jaw_angle - first.jaw_angle).abs() < 1e-5);
    }

    #[test]
    fn test_sample_at_t_duration_returns_last_frame_values() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let sampled = clip.sample_at(clip.total_duration);
        let last = clip.keyframes.last().expect("has keyframes");
        assert!((sampled.jaw_angle - last.jaw_angle).abs() < 1e-5);
    }

    #[test]
    fn test_sample_at_midpoint_interpolates() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("sil-aa", 0.5, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 10.0).expect("synth");
        // At midpoint, jaw should be between 0 and max
        let sampled = clip.sample_at(0.5);
        assert!(sampled.jaw_angle.is_finite());
    }

    // --- apply_coarticulation ---

    #[test]
    fn test_coarticulation_output_length_matches_input() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t-s", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let smoothed = apply_coarticulation(&clip.keyframes, 0.05, 30.0);
        assert_eq!(smoothed.len(), clip.keyframes.len());
    }

    #[test]
    fn test_coarticulation_empty_input_empty_output() {
        let out = apply_coarticulation(&[], 0.1, 30.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_coarticulation_times_preserved() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let smoothed = apply_coarticulation(&clip.keyframes, 0.05, 30.0);
        for (orig, sm) in clip.keyframes.iter().zip(smoothed.iter()) {
            assert!((orig.time - sm.time).abs() < 1e-6);
        }
    }

    // --- smooth_phoneme_keyframes ---

    #[test]
    fn test_smooth_sigma_zero_unchanged() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let smoothed = smooth_phoneme_keyframes(&clip.keyframes, 0.0);
        assert_eq!(smoothed.len(), clip.keyframes.len());
        for (orig, sm) in clip.keyframes.iter().zip(smoothed.iter()) {
            assert!((orig.jaw_angle - sm.jaw_angle).abs() < 1e-6);
        }
    }

    #[test]
    fn test_smooth_sigma_positive_reduces_variance() {
        // Build two extreme keyframes (jaw 0 and jaw 0.3) and smooth
        let kfs = vec![
            PhonemeKeyframe {
                time: 0.0,
                jaw_angle: 0.0,
                expression: vec![0.0],
            },
            PhonemeKeyframe {
                time: 0.1,
                jaw_angle: 0.3,
                expression: vec![1.0],
            },
            PhonemeKeyframe {
                time: 0.2,
                jaw_angle: 0.0,
                expression: vec![0.0],
            },
        ];
        let smoothed = smooth_phoneme_keyframes(&kfs, 1.5);
        // After smoothing, middle frame should be damped
        let raw_range = 0.3_f32;
        let smoothed_range = smoothed.iter().map(|k| k.jaw_angle).fold(0.0_f32, f32::max)
            - smoothed
                .iter()
                .map(|k| k.jaw_angle)
                .fold(f32::MAX, f32::min);
        assert!(smoothed_range < raw_range, "Smoothing should reduce range");
    }

    #[test]
    fn test_smooth_output_length_unchanged() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t-k", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let smoothed = smooth_phoneme_keyframes(&clip.keyframes, 2.0);
        assert_eq!(smoothed.len(), clip.keyframes.len());
    }

    // --- generate_breath_animation ---

    #[test]
    fn test_breath_animation_correct_frame_count() {
        let kfs = generate_breath_animation(1.0, 12.0, 4, 1.0);
        // 1.0s at 30fps = 30 frames
        assert_eq!(kfs.len(), 30);
    }

    #[test]
    fn test_breath_animation_values_in_range() {
        let kfs = generate_breath_animation(2.0, 12.0, 4, 1.0);
        for kf in &kfs {
            // Jaw is non-negative (sin^2 based)
            assert!(kf.jaw_angle >= -1e-6, "jaw_angle should be non-negative");
            assert!(kf.jaw_angle < 1.0, "jaw_angle too large");
        }
    }

    #[test]
    fn test_breath_animation_jaw_period_matches_breath_rate() {
        // 12 breaths/min = 0.2 Hz → one full jaw cycle every 5 s: the jaw is
        // fully open at t = 2.5 s (frame 75) and closed again at t = 5.0 s
        // (frame 150).  The previous `sin²(2πft)` form ran at twice the
        // documented rate and was ~0 at t = 2.5 s.
        let kfs = generate_breath_animation(6.0, 12.0, 4, 1.0);
        assert!(kfs.len() > 150, "expected >150 frames, got {}", kfs.len());
        assert!(kfs[0].jaw_angle.abs() < 1e-6, "jaw should start closed");
        let peak = kfs[75].jaw_angle;
        let trough = kfs[150].jaw_angle;
        assert!(
            (peak - 0.05).abs() < 1e-4,
            "jaw should be fully open at t = 2.5 s, got {peak}"
        );
        assert!(
            trough.abs() < 1e-4,
            "jaw should be closed again at t = 5.0 s, got {trough}"
        );
    }

    #[test]
    fn test_breath_animation_zero_duration_empty() {
        let kfs = generate_breath_animation(0.0, 12.0, 4, 1.0);
        assert!(kfs.is_empty());
    }

    #[test]
    fn test_breath_animation_zero_amplitude() {
        let kfs = generate_breath_animation(1.0, 12.0, 4, 0.0);
        // All jaw angles should be zero (amplitude=0 means no movement)
        for kf in &kfs {
            assert!(kf.jaw_angle.abs() < 1e-6);
        }
    }

    // --- blend_phoneme_with_base ---

    #[test]
    fn test_blend_weight_one_returns_phoneme() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("ae", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 10.0).expect("synth");
        let base_expr = vec![0.0_f32; 10];
        let blended =
            blend_phoneme_with_base(&clip.keyframes, &base_expr, 0.0, 1.0).expect("blend");
        for (orig, bl) in clip.keyframes.iter().zip(blended.iter()) {
            assert!((orig.jaw_angle - bl.jaw_angle).abs() < 1e-5);
        }
    }

    #[test]
    fn test_blend_weight_zero_returns_base() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("ae", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 10.0).expect("synth");
        let base_expr = vec![0.5_f32; 10];
        let base_jaw = 0.1;
        let blended =
            blend_phoneme_with_base(&clip.keyframes, &base_expr, base_jaw, 0.0).expect("blend");
        for bl in &blended {
            assert!((bl.jaw_angle - base_jaw).abs() < 1e-5);
            for (v, b) in bl.expression.iter().zip(base_expr.iter()) {
                assert!((v - b).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_blend_mismatch_error() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("ae", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 10.0).expect("synth");
        let base_expr = vec![0.0_f32; 5]; // wrong length
        let result = blend_phoneme_with_base(&clip.keyframes, &base_expr, 0.0, 0.5);
        assert!(matches!(
            result,
            Err(PhonemeError::ExpressionMismatch { .. })
        ));
    }

    // --- extract_jaw_sequence ---

    #[test]
    fn test_extract_jaw_sequence_length() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let jaw = extract_jaw_sequence(&clip.keyframes);
        assert_eq!(jaw.len(), clip.keyframes.len());
    }

    // --- extract_expression_sequence ---

    #[test]
    fn test_extract_expression_sequence_dimensions() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let exprs = extract_expression_sequence(&clip.keyframes);
        assert_eq!(exprs.len(), clip.keyframes.len());
        for row in &exprs {
            assert_eq!(row.len(), clip.n_expression_coeffs);
        }
    }

    // --- phoneme_clip_stats ---

    #[test]
    fn test_clip_stats_n_events() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let stats = phoneme_clip_stats(&clip);
        assert_eq!(stats.n_events, 3);
    }

    #[test]
    fn test_clip_stats_n_keyframes() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let stats = phoneme_clip_stats(&clip);
        assert_eq!(stats.n_keyframes, clip.keyframes.len());
    }

    #[test]
    fn test_clip_stats_duration() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let stats = phoneme_clip_stats(&clip);
        assert!((stats.total_duration - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_clip_stats_speaking_fraction() {
        let lib = PhonemeLibrary::default_english(10);
        // All non-silent
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let stats = phoneme_clip_stats(&clip);
        assert!((stats.speaking_fraction - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_clip_stats_speaking_fraction_with_silence() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("sil-ae-sil", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let stats = phoneme_clip_stats(&clip);
        // 1 of 3 events is non-silent
        assert!((stats.speaking_fraction - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_clip_stats_max_jaw_positive_for_vowels() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("aa-aa-aa", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let stats = phoneme_clip_stats(&clip);
        assert!(stats.max_jaw_angle > 0.0);
    }

    // --- format_phoneme_stats ---

    #[test]
    fn test_format_phoneme_stats_nonempty() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let stats = phoneme_clip_stats(&clip);
        let s = format_phoneme_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("events"));
    }

    // --- format_phoneme_clip ---

    #[test]
    fn test_format_phoneme_clip_nonempty() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let s = format_phoneme_clip(&clip);
        assert!(!s.is_empty());
        assert!(s.contains("PhonemeClip"));
    }

    // --- Error cases ---

    #[test]
    fn test_error_unknown_phoneme_display() {
        let e = PhonemeError::UnknownPhoneme("xyz".to_string());
        assert!(e.to_string().contains("xyz"));
    }

    #[test]
    fn test_error_empty_sequence_display() {
        let e = PhonemeError::EmptySequence;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_invalid_duration_display() {
        let e = PhonemeError::InvalidDuration(-1.0);
        assert!(e.to_string().contains("-1"));
    }

    #[test]
    fn test_phoneme_symbols_coverage() {
        // Every symbol returned by phoneme_symbols should round-trip through from_phoneme
        let visemes = [
            Viseme::Silence,
            Viseme::Bilabial,
            Viseme::Labiodental,
            Viseme::Dental,
            Viseme::Alveolar,
            Viseme::Palatal,
            Viseme::Velar,
            Viseme::Glottal,
            Viseme::OpenVowel,
            Viseme::MidVowel,
            Viseme::ClosedVowel,
            Viseme::RoundedVowel,
        ];
        for v in &visemes {
            for sym in v.phoneme_symbols() {
                let mapped = Viseme::from_phoneme(sym);
                assert_eq!(
                    mapped.as_ref(),
                    Some(v),
                    "Symbol {sym:?} should map to {v:?}"
                );
            }
        }
    }

    #[test]
    fn test_library_n_expression_coeffs_preserved() {
        let lib = PhonemeLibrary::default_english(15);
        assert_eq!(lib.n_expression_coeffs, 15);
    }

    #[test]
    fn test_jaw_sequence_values_finite() {
        let lib = PhonemeLibrary::default_english(10);
        let events = parse_phoneme_string("p-ae-t-k-aa", 0.1, &lib).expect("parse");
        let clip = synthesize_phoneme_animation(&events, &lib, 30.0).expect("synth");
        let jaw = extract_jaw_sequence(&clip.keyframes);
        for &j in &jaw {
            assert!(j.is_finite());
        }
    }
}
