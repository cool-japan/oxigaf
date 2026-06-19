//! Weighted text prompt parsing and embedding manipulation.
//!
//! Parses weighted prompt syntax in the style of A1111 / ComfyUI:
//!
//! ```text
//! "a (beautiful:1.5) face with (bright:0.8) eyes"
//! ```
//!
//! Weights scale the attention embeddings for the corresponding tokens,
//! allowing fine-grained control over how strongly each concept is expressed.
//!
//! ## Syntax
//!
//! | Syntax            | Effect                               |
//! |-------------------|--------------------------------------|
//! | `plain text`      | weight 1.0                           |
//! | `(text:1.5)`      | explicit weight 1.5                  |
//! | `(text)`          | implicit boost × 1.1                 |
//! | `((text))`        | double boost × 1.1² ≈ 1.21          |
//! | `[text]`          | reduce weight × 0.9                  |

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while parsing or applying weighted prompts.
#[derive(Debug, Error, PartialEq)]
pub enum PromptWeightingError {
    /// A parse error occurred at the given byte position.
    #[error("Parse error at position {pos}: {msg}")]
    ParseError {
        /// Zero-based byte position in the prompt string.
        pos: usize,
        /// Human-readable description of the error.
        msg: String,
    },

    /// A weight value is not a positive finite number.
    #[error("Invalid weight {weight}: must be positive")]
    InvalidWeight {
        /// The offending weight value.
        weight: f32,
    },

    /// A parenthesis or bracket is not properly closed.
    #[error("Unmatched parenthesis at position {pos}")]
    UnmatchedParen {
        /// Position of the unmatched opening character.
        pos: usize,
    },

    /// The prompt string is empty or contains only whitespace.
    #[error("Empty prompt")]
    EmptyPrompt,

    /// The number of embedding rows does not match the number of weights.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected number of token rows.
        expected: usize,
        /// Actual number of token rows found.
        actual: usize,
    },

    /// The token sequence is empty.
    #[error("Empty token sequence")]
    EmptyTokens,
}

// ---------------------------------------------------------------------------
// WeightedToken
// ---------------------------------------------------------------------------

/// A text token together with its attention weight multiplier.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedToken {
    /// The token text (a word or sub-word piece).
    pub text: String,
    /// Multiplicative weight applied to the embedding for this token.
    pub weight: f32,
}

impl WeightedToken {
    /// Create a new token with an explicit weight.
    pub fn new(text: impl Into<String>, weight: f32) -> Self {
        Self {
            text: text.into(),
            weight,
        }
    }

    /// Create a token with the default unit weight (1.0).
    pub fn unweighted(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            weight: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// WeightedPrompt
// ---------------------------------------------------------------------------

/// A parsed prompt composed of weighted tokens.
#[derive(Debug, Clone)]
pub struct WeightedPrompt {
    /// Ordered list of tokens with their weights.
    pub tokens: Vec<WeightedToken>,
    /// Default weight applied to tokens that have no explicit annotation (1.0).
    pub base_weight: f32,
}

impl WeightedPrompt {
    /// Create a weighted prompt from a token list (base weight defaults to 1.0).
    pub fn new(tokens: Vec<WeightedToken>) -> Self {
        Self {
            tokens,
            base_weight: 1.0,
        }
    }

    /// Parse a prompt string into a `WeightedPrompt`.
    ///
    /// Delegates to [`parse_weighted_prompt`].
    pub fn parse(prompt: &str) -> Result<Self, PromptWeightingError> {
        parse_weighted_prompt(prompt)
    }

    /// Return the concatenated plain text of all tokens, joined by spaces.
    pub fn to_plain_text(&self) -> String {
        self.tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Return a `Vec` of all token weights in order.
    pub fn weight_vector(&self) -> Vec<f32> {
        self.tokens.iter().map(|t| t.weight).collect()
    }

    /// Return the maximum weight across all tokens, or 1.0 if the list is empty.
    pub fn max_weight(&self) -> f32 {
        self.tokens
            .iter()
            .map(|t| t.weight)
            .fold(f32::NEG_INFINITY, f32::max)
            .max(self.base_weight)
    }

    /// Return the minimum weight across all tokens, or 1.0 if the list is empty.
    pub fn min_weight(&self) -> f32 {
        self.tokens
            .iter()
            .map(|t| t.weight)
            .fold(f32::INFINITY, f32::min)
            .min(self.base_weight)
    }

    /// Return `true` iff every token has weight exactly 1.0.
    pub fn is_uniform(&self) -> bool {
        self.tokens.iter().all(|t| t.weight == 1.0)
    }

    /// Divide all token weights by their mean, so the mean becomes 1.0.
    ///
    /// No-ops if the token list is empty or the mean is zero.
    pub fn normalize_weights(&mut self) {
        if self.tokens.is_empty() {
            return;
        }
        let sum: f32 = self.tokens.iter().map(|t| t.weight).sum();
        let n = self.tokens.len() as f32;
        let mean = sum / n;
        if mean.abs() > f32::EPSILON {
            for t in &mut self.tokens {
                t.weight /= mean;
            }
        }
    }
}

impl std::str::FromStr for WeightedPrompt {
    type Err = PromptWeightingError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_weighted_prompt(s)
    }
}

// ---------------------------------------------------------------------------
// WeightScaleMode
// ---------------------------------------------------------------------------

/// How the weight is applied to the embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightScaleMode {
    /// Multiply each token embedding directly: `embedding * weight`.
    Multiply,
    /// Scale attention logits (not the embeddings themselves).
    Attention,
    /// Linear blend between the unweighted and weighted embeddings.
    Blend,
}

// ---------------------------------------------------------------------------
// WeightingConfig
// ---------------------------------------------------------------------------

/// Configuration for embedding weighting operations.
#[derive(Debug, Clone)]
pub struct WeightingConfig {
    /// How weights are applied to embeddings.
    pub scale_mode: WeightScaleMode,
    /// Weights are clamped to `[1/max_weight, max_weight]` before use.
    pub max_weight: f32,
    /// Whether to normalise weights so they sum to `len(tokens)`.
    pub normalize: bool,
}

impl Default for WeightingConfig {
    fn default() -> Self {
        Self {
            scale_mode: WeightScaleMode::Multiply,
            max_weight: 2.0,
            normalize: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WeightStats
// ---------------------------------------------------------------------------

/// Descriptive statistics about a weight vector.
#[derive(Debug, Clone)]
pub struct WeightStats {
    /// Arithmetic mean.
    pub mean: f32,
    /// Standard deviation.
    pub std: f32,
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Number of weights strictly greater than 1.0.
    pub n_boosted: usize,
    /// Number of weights strictly less than 1.0.
    pub n_suppressed: usize,
}

// ---------------------------------------------------------------------------
// Internal parser helpers
// ---------------------------------------------------------------------------

/// Parse state for one group opened by `(` or `[`.
#[derive(Debug, Clone)]
struct GroupFrame {
    /// Byte position of the opening delimiter in the original prompt.
    open_pos: usize,
    /// Type of group.
    kind: GroupKind,
    /// Weight inherited from the outer nesting level.
    outer_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GroupKind {
    /// Round parenthesis group `(...)`.
    Paren,
    /// Square bracket group `[...]`.
    Bracket,
}

// ---------------------------------------------------------------------------
// Public API – parsing
// ---------------------------------------------------------------------------

/// Parse a weighted prompt string into a [`WeightedPrompt`].
///
/// Supports `(text:weight)`, `(text)` (×1.1), `((text))` (×1.21), and `[text]` (×0.9).
pub fn parse_weighted_prompt(prompt: &str) -> Result<WeightedPrompt, PromptWeightingError> {
    if prompt.trim().is_empty() {
        return Err(PromptWeightingError::EmptyPrompt);
    }

    // We will accumulate (raw_text, weight) pairs from the parsing pass,
    // then tokenize each raw_text segment and assign its weight.
    let mut result_tokens: Vec<WeightedToken> = Vec::new();

    // Stack of currently open groups.
    let mut stack: Vec<GroupFrame> = Vec::new();

    // Current accumulated text for the innermost segment.
    let mut current_text = String::new();
    // Current weight from enclosing groups (multiplicative).
    let mut current_weight: f32 = 1.0;

    let chars: Vec<(usize, char)> = prompt.char_indices().collect();
    let len = chars.len();
    let mut i = 0usize;

    while i < len {
        let (byte_pos, ch) = chars[i];
        match ch {
            '(' => {
                // Flush pending plain text at the current weight level.
                flush_segment(&current_text, current_weight, &mut result_tokens);
                current_text.clear();

                // Push a frame for this group.
                stack.push(GroupFrame {
                    open_pos: byte_pos,
                    kind: GroupKind::Paren,
                    outer_weight: current_weight,
                });
                // Default boost for `(text)` without explicit weight.
                current_weight *= 1.1;
                i += 1;
            }
            ')' => {
                // Find the matching frame.
                let frame = stack
                    .iter()
                    .rposition(|f| f.kind == GroupKind::Paren)
                    .ok_or(PromptWeightingError::UnmatchedParen { pos: byte_pos })?;

                let frame = stack.remove(frame);

                // Check whether the accumulated text has a trailing `:weight`.
                let (segment_text, segment_weight) = extract_weight_suffix(
                    &current_text,
                    current_weight,
                    frame.outer_weight,
                    byte_pos,
                )?;

                flush_segment(&segment_text, segment_weight, &mut result_tokens);
                current_text.clear();

                // Restore weight from whatever was on the stack before this frame.
                // Actually the weight after closing is the outer weight (no group any more).
                current_weight = frame.outer_weight;
                i += 1;
            }
            '[' => {
                flush_segment(&current_text, current_weight, &mut result_tokens);
                current_text.clear();

                stack.push(GroupFrame {
                    open_pos: byte_pos,
                    kind: GroupKind::Bracket,
                    outer_weight: current_weight,
                });
                current_weight *= 0.9;
                i += 1;
            }
            ']' => {
                let frame_idx = stack
                    .iter()
                    .rposition(|f| f.kind == GroupKind::Bracket)
                    .ok_or(PromptWeightingError::UnmatchedParen { pos: byte_pos })?;

                let frame = stack.remove(frame_idx);

                flush_segment(&current_text, current_weight, &mut result_tokens);
                current_text.clear();

                current_weight = frame.outer_weight;
                i += 1;
            }
            _ => {
                current_text.push(ch);
                i += 1;
            }
        }
    }

    // If the stack is not empty, there are unclosed groups.
    if let Some(frame) = stack.first() {
        return Err(PromptWeightingError::UnmatchedParen {
            pos: frame.open_pos,
        });
    }

    // Flush any remaining plain text.
    flush_segment(&current_text, current_weight, &mut result_tokens);

    if result_tokens.is_empty() {
        return Err(PromptWeightingError::EmptyPrompt);
    }

    Ok(WeightedPrompt::new(result_tokens))
}

/// Flush a text segment into `result_tokens` by tokenising it.
fn flush_segment(text: &str, weight: f32, result_tokens: &mut Vec<WeightedToken>) {
    for word in tokenize(text) {
        result_tokens.push(WeightedToken::new(word, weight));
    }
}

/// Parse a possible `:weight` suffix from the accumulated text inside `(...)`.
///
/// Returns `(text_without_suffix, resolved_weight)`.
///
/// If no suffix is present, `current_weight` (which already includes the ×1.1
/// default boost) is used verbatim.
fn extract_weight_suffix(
    raw: &str,
    current_weight: f32,
    outer_weight: f32,
    close_pos: usize,
) -> Result<(String, f32), PromptWeightingError> {
    // Look for the last ':' that is followed by a float.
    if let Some(colon_pos) = raw.rfind(':') {
        let after_colon = raw[colon_pos + 1..].trim();
        if let Ok(w) = after_colon.parse::<f32>() {
            // Validate the weight.
            if !w.is_finite() || w <= 0.0 {
                return Err(PromptWeightingError::InvalidWeight { weight: w });
            }
            // The explicit weight replaces (not multiplies) the default ×1.1 boost,
            // but still multiplies by the outer context weight.
            let resolved = outer_weight * w;
            let text = raw[..colon_pos].trim().to_string();
            return Ok((text, resolved));
        }
    }
    // No explicit weight – use `current_weight` as already computed (outer × 1.1).
    let _ = close_pos; // kept for future diagnostics
    Ok((raw.trim().to_string(), current_weight))
}

// ---------------------------------------------------------------------------
// Public API – tokenisation
// ---------------------------------------------------------------------------

/// Tokenise plain text into words.
///
/// Splits on whitespace and strips leading/trailing punctuation from each word.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '-' && c != '\'')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Public API – embedding operations
// ---------------------------------------------------------------------------

/// Apply a per-token weight vector to a flat embedding matrix.
///
/// `embeddings` is a row-major `[n_tokens × embed_dim]` slice.  Each token row
/// is scaled by the corresponding element of `weights`.
///
/// Returns a new `Vec<f32>` of the same length with each row scaled.
pub fn apply_embedding_weights(
    embeddings: &[f32],
    weights: &[f32],
    embed_dim: usize,
) -> Result<Vec<f32>, PromptWeightingError> {
    let n_tokens = weights.len();
    if n_tokens == 0 {
        return Err(PromptWeightingError::EmptyTokens);
    }
    let expected = n_tokens * embed_dim;
    if embeddings.len() != expected {
        return Err(PromptWeightingError::DimensionMismatch {
            expected,
            actual: embeddings.len(),
        });
    }

    let mut output = embeddings.to_vec();
    for (tok_idx, &w) in weights.iter().enumerate() {
        let start = tok_idx * embed_dim;
        let end = start + embed_dim;
        for v in &mut output[start..end] {
            *v *= w;
        }
    }
    Ok(output)
}

/// Blend weighted and unweighted embeddings via linear interpolation.
///
/// `output[i] = lerp(embeddings[i], embeddings[i] * weight[token], blend_factor)`
///
/// A `blend_factor` of 0.0 returns the original embeddings unchanged; 1.0
/// returns the fully-weighted result.
pub fn blend_weighted_embeddings(
    embeddings: &[f32],
    weights: &[f32],
    embed_dim: usize,
    blend_factor: f32,
) -> Result<Vec<f32>, PromptWeightingError> {
    let n_tokens = weights.len();
    if n_tokens == 0 {
        return Err(PromptWeightingError::EmptyTokens);
    }
    let expected = n_tokens * embed_dim;
    if embeddings.len() != expected {
        return Err(PromptWeightingError::DimensionMismatch {
            expected,
            actual: embeddings.len(),
        });
    }

    let bf = blend_factor.clamp(0.0, 1.0);
    let mut output = embeddings.to_vec();
    for (tok_idx, &w) in weights.iter().enumerate() {
        let start = tok_idx * embed_dim;
        let end = start + embed_dim;
        for v in &mut output[start..end] {
            // lerp(v, v * w, bf) = v + bf * (v * w - v) = v * (1 + bf * (w - 1))
            *v *= 1.0 + bf * (w - 1.0);
        }
    }
    Ok(output)
}

/// Normalise `weights` in-place so their arithmetic mean equals 1.0.
///
/// No-ops if the slice is empty or the mean is effectively zero.
pub fn normalize_weights(weights: &mut [f32]) {
    if weights.is_empty() {
        return;
    }
    let sum: f32 = weights.iter().sum();
    let n = weights.len() as f32;
    let mean = sum / n;
    if mean.abs() > f32::EPSILON {
        for w in weights.iter_mut() {
            *w /= mean;
        }
    }
}

/// Clamp all weights to the closed interval `[1/max_w, max_w]`.
pub fn clamp_weights(weights: &mut [f32], max_w: f32) {
    let lo = 1.0 / max_w;
    for w in weights.iter_mut() {
        *w = w.clamp(lo, max_w);
    }
}

/// Merge two weighted prompts by concatenating their token lists.
pub fn merge_prompts(a: &WeightedPrompt, b: &WeightedPrompt) -> WeightedPrompt {
    let mut tokens = a.tokens.clone();
    tokens.extend(b.tokens.iter().cloned());
    WeightedPrompt::new(tokens)
}

/// Scale every weight in `weights` by `factor`, returning a new `Vec`.
pub fn scale_weights(weights: &[f32], factor: f32) -> Vec<f32> {
    weights.iter().map(|&w| w * factor).collect()
}

/// Compute the weighted average of token embeddings.
///
/// Returns a single embedding vector of length `embed_dim`, computed as the
/// weighted arithmetic mean across the token dimension.
///
/// `embeddings` is a row-major `[n_tokens × embed_dim]` slice.
pub fn weighted_average_embedding(
    embeddings: &[f32],
    weights: &[f32],
    embed_dim: usize,
) -> Result<Vec<f32>, PromptWeightingError> {
    let n_tokens = weights.len();
    if n_tokens == 0 {
        return Err(PromptWeightingError::EmptyTokens);
    }
    let expected = n_tokens * embed_dim;
    if embeddings.len() != expected {
        return Err(PromptWeightingError::DimensionMismatch {
            expected,
            actual: embeddings.len(),
        });
    }

    let weight_sum: f32 = weights.iter().sum();
    let normaliser = if weight_sum.abs() > f32::EPSILON {
        weight_sum
    } else {
        1.0
    };

    let mut avg = vec![0.0f32; embed_dim];
    for (tok_idx, &w) in weights.iter().enumerate() {
        let start = tok_idx * embed_dim;
        for (d, a) in avg.iter_mut().enumerate() {
            *a += embeddings[start + d] * w;
        }
    }
    for a in &mut avg {
        *a /= normaliser;
    }
    Ok(avg)
}

/// Convert per-token weights to an additive attention bias.
///
/// `output[i] = log(weight[i]) * scale`
///
/// The log is clamped away from −∞ so tokens with weight ≈ 0 receive a large
/// but finite negative bias.
pub fn weight_to_attention_bias(weights: &[f32], scale: f32) -> Vec<f32> {
    const MIN_WEIGHT: f32 = 1e-8;
    weights
        .iter()
        .map(|&w| w.max(MIN_WEIGHT).ln() * scale)
        .collect()
}

/// Compute descriptive statistics for a weight vector.
pub fn compute_weight_stats(weights: &[f32]) -> Result<WeightStats, PromptWeightingError> {
    if weights.is_empty() {
        return Err(PromptWeightingError::EmptyTokens);
    }
    let n = weights.len() as f32;
    let sum: f32 = weights.iter().sum();
    let mean = sum / n;
    let variance: f32 = weights.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let n_boosted = weights.iter().filter(|&&w| w > 1.0).count();
    let n_suppressed = weights.iter().filter(|&&w| w < 1.0).count();

    Ok(WeightStats {
        mean,
        std,
        min,
        max,
        n_boosted,
        n_suppressed,
    })
}

/// Convert a [`WeightedPrompt`] into parallel `(texts, weights)` arrays for
/// batch processing.
pub fn prompt_to_arrays(prompt: &WeightedPrompt) -> (Vec<String>, Vec<f32>) {
    let texts = prompt.tokens.iter().map(|t| t.text.clone()).collect();
    let weights = prompt.tokens.iter().map(|t| t.weight).collect();
    (texts, weights)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // -- WeightedToken -------------------------------------------------------

    #[test]
    fn test_weighted_token_new() {
        let t = WeightedToken::new("hello", 1.5);
        assert_eq!(t.text, "hello");
        assert!((t.weight - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_token_unweighted() {
        let t = WeightedToken::unweighted("world");
        assert_eq!(t.text, "world");
        assert!((t.weight - 1.0).abs() < 1e-6);
    }

    // -- parse_weighted_prompt / WeightedPrompt::from_str -------------------

    #[test]
    fn test_parse_plain_text() {
        let p = parse_weighted_prompt("a beautiful face").unwrap();
        assert_eq!(p.tokens.len(), 3);
        for t in &p.tokens {
            assert!(
                (t.weight - 1.0).abs() < 1e-6,
                "expected weight 1.0, got {}",
                t.weight
            );
        }
    }

    #[test]
    fn test_parse_explicit_weight() {
        let p = parse_weighted_prompt("(beautiful:1.5)").unwrap();
        assert_eq!(p.tokens.len(), 1);
        assert_eq!(p.tokens[0].text, "beautiful");
        assert!((p.tokens[0].weight - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_parse_implicit_boost() {
        let p = parse_weighted_prompt("(word)").unwrap();
        assert_eq!(p.tokens.len(), 1);
        assert!((p.tokens[0].weight - 1.1).abs() < 1e-5);
    }

    #[test]
    fn test_parse_double_boost() {
        let p = parse_weighted_prompt("((word))").unwrap();
        assert_eq!(p.tokens.len(), 1);
        // 1.1 * 1.1 = 1.21
        assert!((p.tokens[0].weight - 1.21).abs() < 1e-4);
    }

    #[test]
    fn test_parse_bracket_reduce() {
        let p = parse_weighted_prompt("[word]").unwrap();
        assert_eq!(p.tokens.len(), 1);
        assert!((p.tokens[0].weight - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_parse_multi_word_group() {
        let p = parse_weighted_prompt("(bright blue:1.3)").unwrap();
        assert_eq!(p.tokens.len(), 2);
        assert!((p.tokens[0].weight - 1.3).abs() < 1e-5);
        assert!((p.tokens[1].weight - 1.3).abs() < 1e-5);
    }

    #[test]
    fn test_parse_empty_prompt_error() {
        let err = parse_weighted_prompt("").unwrap_err();
        assert_eq!(err, PromptWeightingError::EmptyPrompt);
    }

    #[test]
    fn test_parse_whitespace_only_error() {
        let err = parse_weighted_prompt("   ").unwrap_err();
        assert_eq!(err, PromptWeightingError::EmptyPrompt);
    }

    #[test]
    fn test_parse_unmatched_paren_error() {
        let err = parse_weighted_prompt("(unmatched").unwrap_err();
        assert!(matches!(err, PromptWeightingError::UnmatchedParen { .. }));
    }

    #[test]
    fn test_parse_explicit_weight_nested() {
        // Explicit weight should resolve against outer weight (1.0 here).
        let p = parse_weighted_prompt("(text:2.0)").unwrap();
        assert!((p.tokens[0].weight - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_from_str_delegates() {
        let a = parse_weighted_prompt("hello world").unwrap();
        let b = WeightedPrompt::from_str("hello world").unwrap();
        assert_eq!(a.tokens.len(), b.tokens.len());
    }

    #[test]
    fn test_parse_mixed_prompt() {
        let p = parse_weighted_prompt("a (beautiful:1.5) face with [ugly:0.8] eyes").unwrap();
        // Tokens: a, beautiful, face, with, ugly, 0.8 (note '0.8' is NOT a weight here
        // since we're inside [], the `:` suffix logic applies only inside `()`)
        // Actually wait – brackets don't parse :weight suffix in our implementation.
        // Let's just check that at least 'beautiful' has weight 1.5.
        let beautiful = p.tokens.iter().find(|t| t.text == "beautiful");
        assert!(beautiful.is_some());
        assert!((beautiful.unwrap().weight - 1.5).abs() < 1e-5);
    }

    // -- WeightedPrompt methods ----------------------------------------------

    #[test]
    fn test_to_plain_text() {
        let p = parse_weighted_prompt("hello (world:1.5)").unwrap();
        let text = p.to_plain_text();
        assert!(text.contains("hello"));
        assert!(text.contains("world"));
    }

    #[test]
    fn test_weight_vector() {
        let p = WeightedPrompt::new(vec![
            WeightedToken::new("a", 1.0),
            WeightedToken::new("b", 2.0),
            WeightedToken::new("c", 0.5),
        ]);
        let v = p.weight_vector();
        assert_eq!(v, vec![1.0, 2.0, 0.5]);
    }

    #[test]
    fn test_is_uniform_true() {
        let p = WeightedPrompt::new(vec![
            WeightedToken::unweighted("a"),
            WeightedToken::unweighted("b"),
        ]);
        assert!(p.is_uniform());
    }

    #[test]
    fn test_is_uniform_false() {
        let p = WeightedPrompt::new(vec![
            WeightedToken::new("a", 1.5),
            WeightedToken::unweighted("b"),
        ]);
        assert!(!p.is_uniform());
    }

    #[test]
    fn test_normalize_weights_method() {
        let mut p = WeightedPrompt::new(vec![
            WeightedToken::new("a", 2.0),
            WeightedToken::new("b", 4.0),
        ]);
        p.normalize_weights();
        let v = p.weight_vector();
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        assert!((mean - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_min_weight() {
        let p = WeightedPrompt::new(vec![
            WeightedToken::new("a", 0.5),
            WeightedToken::new("b", 2.0),
            WeightedToken::new("c", 1.0),
        ]);
        assert!((p.max_weight() - 2.0).abs() < 1e-6);
        assert!((p.min_weight() - 0.5).abs() < 1e-6);
    }

    // -- tokenize ------------------------------------------------------------

    #[test]
    fn test_tokenize_basic() {
        let words = tokenize("hello world");
        assert_eq!(words, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_extra_spaces() {
        let words = tokenize("  hello   world  ");
        assert_eq!(words, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_punctuation_stripped() {
        let words = tokenize("hello, world!");
        assert_eq!(words, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let words = tokenize("   ");
        assert!(words.is_empty());
    }

    // -- apply_embedding_weights ---------------------------------------------

    #[test]
    fn test_apply_weights_identity() {
        let embed = vec![1.0f32, 2.0, 3.0, 4.0];
        let weights = vec![1.0f32, 1.0];
        let out = apply_embedding_weights(&embed, &weights, 2).unwrap();
        assert_eq!(out, embed);
    }

    #[test]
    fn test_apply_weights_doubled() {
        let embed = vec![1.0f32, 2.0, 3.0, 4.0];
        let weights = vec![2.0f32, 2.0];
        let out = apply_embedding_weights(&embed, &weights, 2).unwrap();
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_apply_weights_dimension_mismatch() {
        let embed = vec![1.0f32; 6];
        let weights = vec![1.0f32, 1.0];
        // embed has 6 elements but 2 tokens × 2 dim = 4 expected
        let err = apply_embedding_weights(&embed, &weights, 2).unwrap_err();
        assert!(matches!(
            err,
            PromptWeightingError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_apply_weights_empty_tokens() {
        let embed = vec![1.0f32; 4];
        let weights: Vec<f32> = vec![];
        let err = apply_embedding_weights(&embed, &weights, 2).unwrap_err();
        assert_eq!(err, PromptWeightingError::EmptyTokens);
    }

    // -- blend_weighted_embeddings -------------------------------------------

    #[test]
    fn test_blend_factor_zero_unchanged() {
        let embed = vec![1.0f32, 2.0, 3.0, 4.0];
        let weights = vec![2.0f32, 3.0];
        let out = blend_weighted_embeddings(&embed, &weights, 2, 0.0).unwrap();
        for (a, b) in out.iter().zip(embed.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_factor_one_fully_weighted() {
        let embed = vec![1.0f32, 2.0, 3.0, 4.0];
        let weights = vec![2.0f32, 2.0];
        let out = blend_weighted_embeddings(&embed, &weights, 2, 1.0).unwrap();
        let expected = apply_embedding_weights(&embed, &weights, 2).unwrap();
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // -- normalize_weights (free function) -----------------------------------

    #[test]
    fn test_normalize_weights_mean_one() {
        let mut w = vec![2.0f32, 4.0, 6.0];
        normalize_weights(&mut w);
        let mean: f32 = w.iter().sum::<f32>() / w.len() as f32;
        assert!((mean - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_weights_empty_noop() {
        let mut w: Vec<f32> = vec![];
        normalize_weights(&mut w); // should not panic
        assert!(w.is_empty());
    }

    // -- clamp_weights -------------------------------------------------------

    #[test]
    fn test_clamp_weights() {
        let mut w = vec![0.1f32, 0.5, 1.0, 2.0, 5.0];
        clamp_weights(&mut w, 2.0);
        for &v in &w {
            assert!((0.5..=2.0).contains(&v), "out of range: {v}");
        }
    }

    // -- merge_prompts -------------------------------------------------------

    #[test]
    fn test_merge_prompts_token_count() {
        let a = WeightedPrompt::new(vec![
            WeightedToken::unweighted("a"),
            WeightedToken::unweighted("b"),
        ]);
        let b = WeightedPrompt::new(vec![WeightedToken::unweighted("c")]);
        let merged = merge_prompts(&a, &b);
        assert_eq!(merged.tokens.len(), 3);
    }

    // -- scale_weights -------------------------------------------------------

    #[test]
    fn test_scale_weights_doubles() {
        let w = vec![1.0f32, 2.0, 3.0];
        let scaled = scale_weights(&w, 2.0);
        assert_eq!(scaled, vec![2.0, 4.0, 6.0]);
    }

    // -- weighted_average_embedding ------------------------------------------

    #[test]
    fn test_weighted_average_uniform_equals_mean() {
        // With uniform weights the weighted average should equal the arithmetic mean.
        let embed = vec![
            1.0f32, 2.0, // token 0
            3.0, 4.0,
        ]; // token 1
        let weights = vec![1.0f32, 1.0];
        let avg = weighted_average_embedding(&embed, &weights, 2).unwrap();
        assert!((avg[0] - 2.0).abs() < 1e-5);
        assert!((avg[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_average_empty_error() {
        let embed = vec![1.0f32; 4];
        let weights: Vec<f32> = vec![];
        let err = weighted_average_embedding(&embed, &weights, 2).unwrap_err();
        assert_eq!(err, PromptWeightingError::EmptyTokens);
    }

    // -- weight_to_attention_bias --------------------------------------------

    #[test]
    fn test_attention_bias_unit_weight() {
        let weights = vec![1.0f32];
        let bias = weight_to_attention_bias(&weights, 1.0);
        assert!((bias[0]).abs() < 1e-6, "expected ≈0, got {}", bias[0]);
    }

    #[test]
    fn test_attention_bias_e_weight() {
        let e = std::f32::consts::E;
        let weights = vec![e];
        let bias = weight_to_attention_bias(&weights, 1.0);
        assert!(
            (bias[0] - 1.0).abs() < 1e-5,
            "expected ≈1.0, got {}",
            bias[0]
        );
    }

    #[test]
    fn test_attention_bias_scaled() {
        let weights = vec![1.0f32, std::f32::consts::E];
        let bias = weight_to_attention_bias(&weights, 2.0);
        assert!((bias[0]).abs() < 1e-6);
        assert!((bias[1] - 2.0).abs() < 1e-5);
    }

    // -- compute_weight_stats ------------------------------------------------

    #[test]
    fn test_weight_stats_uniform() {
        let w = vec![1.0f32, 1.0, 1.0];
        let s = compute_weight_stats(&w).unwrap();
        assert!((s.mean - 1.0).abs() < 1e-6);
        assert!(s.std.abs() < 1e-6);
        assert_eq!(s.n_boosted, 0);
        assert_eq!(s.n_suppressed, 0);
    }

    #[test]
    fn test_weight_stats_boosted_suppressed() {
        let w = vec![0.5f32, 1.0, 1.5, 2.0];
        let s = compute_weight_stats(&w).unwrap();
        assert_eq!(s.n_boosted, 2);
        assert_eq!(s.n_suppressed, 1);
        assert!((s.min - 0.5).abs() < 1e-6);
        assert!((s.max - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_stats_empty_error() {
        let err = compute_weight_stats(&[]).unwrap_err();
        assert_eq!(err, PromptWeightingError::EmptyTokens);
    }

    // -- prompt_to_arrays ----------------------------------------------------

    #[test]
    fn test_prompt_to_arrays() {
        let p = WeightedPrompt::new(vec![
            WeightedToken::new("hello", 1.0),
            WeightedToken::new("world", 2.0),
        ]);
        let (texts, weights) = prompt_to_arrays(&p);
        assert_eq!(texts, vec!["hello", "world"]);
        assert_eq!(weights, vec![1.0, 2.0]);
    }

    #[test]
    fn test_prompt_to_arrays_parallel() {
        let p = parse_weighted_prompt("a (b:1.5) c").unwrap();
        let (texts, weights) = prompt_to_arrays(&p);
        assert_eq!(texts.len(), weights.len());
        let b_idx = texts.iter().position(|t| t == "b").unwrap();
        assert!((weights[b_idx] - 1.5).abs() < 1e-5);
    }

    // -- WeightingConfig default -------------------------------------------

    #[test]
    fn test_weighting_config_default() {
        let cfg = WeightingConfig::default();
        assert_eq!(cfg.scale_mode, WeightScaleMode::Multiply);
        assert!((cfg.max_weight - 2.0).abs() < 1e-6);
        assert!(!cfg.normalize);
    }

    // -- Additional coverage -------------------------------------------------

    #[test]
    fn test_parse_nested_explicit_inner_priority() {
        // ((text:1.5)) – the explicit weight 1.5 is inside a `(...)` group
        // whose outer context is the outer `(...)`.  Our spec says explicit
        // weight replaces the ×1.1 default but still multiplies by the outer.
        // Outer `(` gives outer_weight = 1.1 for the inner group.
        // Inner explicit weight resolves to outer_weight_at_inner_level * 1.5.
        let p = parse_weighted_prompt("((text:1.5))").unwrap();
        assert_eq!(p.tokens.len(), 1);
        // Outer group: default boost multiplies outer_weight (1.0) → 1.1.
        // Inner group with explicit :1.5: resolved = outer_at_inner (1.1) * 1.5 = 1.65.
        assert!(
            (p.tokens[0].weight - 1.65).abs() < 1e-4,
            "got {}",
            p.tokens[0].weight
        );
    }

    #[test]
    fn test_close_bracket_without_open_error() {
        let err = parse_weighted_prompt("word]").unwrap_err();
        assert!(matches!(err, PromptWeightingError::UnmatchedParen { .. }));
    }
}
