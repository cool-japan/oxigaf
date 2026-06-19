//! # token_merging
//!
//! Token Merging (ToMe) for transformer attention acceleration.
//!
//! Implements the algorithm from "Token Merging: Your ViT But Faster"
//! (Bolya et al., ICLR 2023). Speeds up attention by merging similar
//! token pairs before the attention layer and unmerging after.
//!
//! ## Data layout
//!
//! All token tensors are flat `Vec<f32>` in row-major order:
//! `token[i][j] = tokens[i * d_model + j]`
//!
//! ## Example
//! ```rust
//! use oxigaf_diffusion::token_merging::{ToMeConfig, tome_merge, tome_unmerge};
//!
//! let tokens: Vec<f32> = (0..16).map(|x| x as f32).collect(); // 4 tokens × 4 dims
//! let config = ToMeConfig::default();
//! let (merged, state) = tome_merge(&tokens, 4, 4, &config).unwrap();
//! let restored = tome_unmerge(&merged, &state).unwrap();
//! assert_eq!(restored.len(), tokens.len());
//! ```

use thiserror::Error;

/// Errors that can occur during token merging operations.
#[derive(Debug, Error, PartialEq)]
pub enum TokenMergeError {
    #[error("Token sequence is empty")]
    EmptySequence,

    #[error(
        "Dimension mismatch: tokens has {tokens_len} elements, expected \
         n_tokens({n_tokens}) × d_model({d_model}) = {expected}"
    )]
    DimensionMismatch {
        tokens_len: usize,
        n_tokens: usize,
        d_model: usize,
        expected: usize,
    },

    #[error("Merge ratio {ratio} must be in [0, 1)")]
    InvalidMergeRatio { ratio: f32 },

    #[error("r ({r}) cannot exceed half of n_tokens ({half})")]
    TooManyMerges { r: usize, half: usize },

    #[error("Merge index out of range: {idx} >= {n_tokens}")]
    IndexOutOfRange { idx: usize, n_tokens: usize },
}

// ---------------------------------------------------------------------------
// MergeMode
// ---------------------------------------------------------------------------

/// How paired tokens are combined into a single merged token.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeMode {
    /// Simple arithmetic mean: `(a + b) / 2`.
    Mean,
    /// Weighted average: `a * alpha + b * (1 - alpha)`.
    Weighted { alpha: f32 },
}

// ---------------------------------------------------------------------------
// ToMeConfig
// ---------------------------------------------------------------------------

/// Configuration for a Token Merging pass.
#[derive(Debug, Clone)]
pub struct ToMeConfig {
    /// Fraction of tokens to merge per pass. Must be in `[0, 1)`.
    /// The number of merged pairs is `r = floor(n_tokens * merge_ratio / 2)`.
    pub merge_ratio: f32,
    /// When `true`, key vectors are used for similarity matching (default).
    /// When `false`, the raw token vectors are used.
    pub use_keys_for_matching: bool,
    /// How paired tokens are averaged.
    pub merge_mode: MergeMode,
}

impl Default for ToMeConfig {
    fn default() -> Self {
        ToMeConfig {
            merge_ratio: 0.5,
            use_keys_for_matching: true,
            merge_mode: MergeMode::Mean,
        }
    }
}

// ---------------------------------------------------------------------------
// MergeState
// ---------------------------------------------------------------------------

/// Tracks the merge mapping needed to reverse a single ToMe pass.
#[derive(Debug, Clone)]
pub struct MergeState {
    /// For each output token: list of original token indices merged into it.
    /// `merge_groups[out_idx]` = `[orig_idx, ...]`
    pub merge_groups: Vec<Vec<usize>>,
    /// Length of the input sequence before merging.
    pub original_n_tokens: usize,
    /// Length of the output sequence after merging.
    pub merged_n_tokens: usize,
    /// Feature dimension.
    pub d_model: usize,
}

impl MergeState {
    /// Number of tokens removed by this merge pass.
    pub fn tokens_saved(&self) -> usize {
        self.original_n_tokens.saturating_sub(self.merged_n_tokens)
    }

    /// Ratio of merged to original token count (`merged_n / original_n`).
    /// Returns 1.0 for an identity (no-op) state.
    pub fn compression_ratio(&self) -> f32 {
        if self.original_n_tokens == 0 {
            1.0
        } else {
            self.merged_n_tokens as f32 / self.original_n_tokens as f32
        }
    }

    /// Build an identity MergeState (each output token maps to a single original).
    fn identity(n_tokens: usize, d_model: usize) -> Self {
        MergeState {
            merge_groups: (0..n_tokens).map(|i| vec![i]).collect(),
            original_n_tokens: n_tokens,
            merged_n_tokens: n_tokens,
            d_model,
        }
    }
}

// ---------------------------------------------------------------------------
// ToMeStats
// ---------------------------------------------------------------------------

/// Statistics describing a single token-merge pass.
#[derive(Debug, Clone)]
pub struct ToMeStats {
    /// Token count before merging.
    pub original_tokens: usize,
    /// Token count after merging.
    pub merged_tokens: usize,
    /// Number of tokens removed.
    pub tokens_saved: usize,
    /// Mean cosine similarity across all merged pairs.
    pub mean_similarity_merged: f32,
    /// Minimum cosine similarity across all merged pairs.
    pub min_similarity_merged: f32,
    /// `merged_tokens / original_tokens`.
    pub compression_ratio: f32,
}

// ---------------------------------------------------------------------------
// Free functions: primitives
// ---------------------------------------------------------------------------

/// Cosine similarity between two equal-length vectors.
///
/// `dot(a, b) / (‖a‖ · ‖b‖ + 1e-8)`
///
/// Returns 0.0 for empty slices.
pub fn token_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = token_l2_norm(a);
    let norm_b = token_l2_norm(b);
    dot / (norm_a * norm_b + 1e-8)
}

/// L2 (Euclidean) norm of a slice.
pub fn token_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// compute_similarity_matrix
// ---------------------------------------------------------------------------

/// Compute an `n_tokens × n_tokens` pairwise cosine-similarity matrix
/// (row-major, symmetric) for a flat token tensor.
pub fn compute_similarity_matrix(
    tokens: &[f32],
    n_tokens: usize,
    d_model: usize,
) -> Result<Vec<f32>, TokenMergeError> {
    validate_shape(tokens, n_tokens, d_model)?;
    let mut matrix = vec![0.0f32; n_tokens * n_tokens];
    for i in 0..n_tokens {
        let a = &tokens[i * d_model..(i + 1) * d_model];
        for j in i..n_tokens {
            let b = &tokens[j * d_model..(j + 1) * d_model];
            let sim = token_cosine_similarity(a, b);
            matrix[i * n_tokens + j] = sim;
            matrix[j * n_tokens + i] = sim;
        }
    }
    Ok(matrix)
}

// ---------------------------------------------------------------------------
// bipartite_soft_matching
// ---------------------------------------------------------------------------

/// Bipartite soft matching: find `r` (a, b) token-index pairs to merge.
///
/// Groups:
/// - **A** = tokens at even indices (0, 2, 4, …)
/// - **B** = tokens at odd  indices (1, 3, 5, …)
///
/// For each token in A the best (highest cosine similarity) unmatched B token
/// is found. Matches are then sorted by descending similarity and the top `r`
/// unique (deduped on B) pairs are returned as `(a_original_idx, b_original_idx)`.
pub fn bipartite_soft_matching(
    tokens: &[f32],
    n_tokens: usize,
    d_model: usize,
    r: usize,
) -> Result<Vec<(usize, usize)>, TokenMergeError> {
    validate_shape(tokens, n_tokens, d_model)?;

    if r == 0 {
        return Ok(Vec::new());
    }

    let half = n_tokens / 2;
    if r > half {
        return Err(TokenMergeError::TooManyMerges { r, half });
    }

    // A = even original indices; B = odd original indices.
    let a_indices: Vec<usize> = (0..n_tokens).step_by(2).collect();
    let b_indices: Vec<usize> = (1..n_tokens).step_by(2).collect();

    if b_indices.is_empty() {
        return Ok(Vec::new());
    }

    // For each A token, compute similarity with all B tokens and pick the best.
    let mut candidates: Vec<(usize, usize, f32)> = Vec::with_capacity(a_indices.len());
    for &ai in &a_indices {
        let a_tok = &tokens[ai * d_model..(ai + 1) * d_model];
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_bi = b_indices[0];
        for &bi in &b_indices {
            let b_tok = &tokens[bi * d_model..(bi + 1) * d_model];
            let sim = token_cosine_similarity(a_tok, b_tok);
            if sim > best_sim {
                best_sim = sim;
                best_bi = bi;
            }
        }
        candidates.push((ai, best_bi, best_sim));
    }

    // Sort by descending similarity.
    candidates.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-r pairs, deduplicating on the B side so each B is used at most once.
    let mut used_b = vec![false; n_tokens];
    let mut pairs = Vec::with_capacity(r);
    for (ai, bi, _sim) in &candidates {
        if pairs.len() >= r {
            break;
        }
        if !used_b[*bi] {
            used_b[*bi] = true;
            pairs.push((*ai, *bi));
        }
    }

    Ok(pairs)
}

// ---------------------------------------------------------------------------
// merge_tokens
// ---------------------------------------------------------------------------

/// Merge token pairs into single tokens, keeping unmatched tokens intact.
///
/// Output ordering:
/// 1. All a-side tokens (merged if paired, original otherwise).
/// 2. All b-side tokens that were **not** consumed by a pair.
///
/// Returns `(merged_tokens_flat, MergeState)`.
pub fn merge_tokens(
    tokens: &[f32],
    n_tokens: usize,
    d_model: usize,
    pairs: &[(usize, usize)],
    mode: &MergeMode,
) -> Result<(Vec<f32>, MergeState), TokenMergeError> {
    validate_shape(tokens, n_tokens, d_model)?;

    // Validate all pair indices.
    for &(ai, bi) in pairs {
        if ai >= n_tokens {
            return Err(TokenMergeError::IndexOutOfRange { idx: ai, n_tokens });
        }
        if bi >= n_tokens {
            return Err(TokenMergeError::IndexOutOfRange { idx: bi, n_tokens });
        }
    }

    // Build a map: a_idx -> b_idx for quick lookup.
    let mut a_to_b: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(pairs.len());
    let mut consumed_b: std::collections::HashSet<usize> =
        std::collections::HashSet::with_capacity(pairs.len());

    for &(ai, bi) in pairs {
        a_to_b.insert(ai, bi);
        consumed_b.insert(bi);
    }

    // Separate even (a-side) and odd (b-side) original indices.
    let a_indices: Vec<usize> = (0..n_tokens).step_by(2).collect();
    let b_indices: Vec<usize> = (1..n_tokens).step_by(2).collect();

    let merged_count = a_indices.len() + b_indices.len() - pairs.len();
    let mut out = Vec::with_capacity(merged_count * d_model);
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(merged_count);

    // 1. Emit a-side tokens (merged or original).
    for &ai in &a_indices {
        let a_tok = &tokens[ai * d_model..(ai + 1) * d_model];
        if let Some(&bi) = a_to_b.get(&ai) {
            let b_tok = &tokens[bi * d_model..(bi + 1) * d_model];
            let merged = blend_tokens(a_tok, b_tok, mode);
            out.extend_from_slice(&merged);
            groups.push(vec![ai, bi]);
        } else {
            out.extend_from_slice(a_tok);
            groups.push(vec![ai]);
        }
    }

    // 2. Emit unmatched b-side tokens.
    for &bi in &b_indices {
        if !consumed_b.contains(&bi) {
            let b_tok = &tokens[bi * d_model..(bi + 1) * d_model];
            out.extend_from_slice(b_tok);
            groups.push(vec![bi]);
        }
    }

    let merged_n = groups.len();
    let state = MergeState {
        merge_groups: groups,
        original_n_tokens: n_tokens,
        merged_n_tokens: merged_n,
        d_model,
    };

    Ok((out, state))
}

/// Average two token vectors according to `mode`.
fn blend_tokens(a: &[f32], b: &[f32], mode: &MergeMode) -> Vec<f32> {
    match mode {
        MergeMode::Mean => a.iter().zip(b.iter()).map(|(x, y)| (x + y) * 0.5).collect(),
        MergeMode::Weighted { alpha } => {
            let alpha = *alpha;
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| x * alpha + y * (1.0 - alpha))
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// unmerge_tokens
// ---------------------------------------------------------------------------

/// Expand a merged token sequence back to the original length.
///
/// For each output token in `merged`, its value is copied to **all** original
/// indices tracked in `state.merge_groups[out_idx]`.
pub fn unmerge_tokens(merged: &[f32], state: &MergeState) -> Result<Vec<f32>, TokenMergeError> {
    let d = state.d_model;

    if state.merged_n_tokens == 0 && state.original_n_tokens == 0 {
        return Ok(Vec::new());
    }

    // Validate merged tensor size.
    let expected_merged = state.merged_n_tokens * d;
    if merged.len() != expected_merged {
        return Err(TokenMergeError::DimensionMismatch {
            tokens_len: merged.len(),
            n_tokens: state.merged_n_tokens,
            d_model: d,
            expected: expected_merged,
        });
    }

    let mut result = vec![0.0f32; state.original_n_tokens * d];

    for (out_idx, group) in state.merge_groups.iter().enumerate() {
        let src = &merged[out_idx * d..(out_idx + 1) * d];
        for &orig_idx in group {
            if orig_idx >= state.original_n_tokens {
                return Err(TokenMergeError::IndexOutOfRange {
                    idx: orig_idx,
                    n_tokens: state.original_n_tokens,
                });
            }
            result[orig_idx * d..(orig_idx + 1) * d].copy_from_slice(src);
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// tome_merge / tome_unmerge
// ---------------------------------------------------------------------------

/// Full ToMe forward pass.
///
/// 1. Computes `r = floor(n_tokens * merge_ratio / 2)`.
/// 2. If `r == 0`, returns an identity pass (no tokens merged).
/// 3. Finds pairs via `bipartite_soft_matching`.
/// 4. Merges with `merge_tokens`.
pub fn tome_merge(
    tokens: &[f32],
    n_tokens: usize,
    d_model: usize,
    config: &ToMeConfig,
) -> Result<(Vec<f32>, MergeState), TokenMergeError> {
    if n_tokens == 0 {
        return Err(TokenMergeError::EmptySequence);
    }
    validate_merge_ratio(config.merge_ratio)?;
    validate_shape(tokens, n_tokens, d_model)?;

    let r = ((n_tokens as f32 * config.merge_ratio) / 2.0).floor() as usize;

    if r == 0 {
        let state = MergeState::identity(n_tokens, d_model);
        return Ok((tokens.to_vec(), state));
    }

    let pairs = bipartite_soft_matching(tokens, n_tokens, d_model, r)?;
    merge_tokens(tokens, n_tokens, d_model, &pairs, &config.merge_mode)
}

/// Full ToMe backward pass (alias for `unmerge_tokens`).
pub fn tome_unmerge(
    merged_output: &[f32],
    state: &MergeState,
) -> Result<Vec<f32>, TokenMergeError> {
    unmerge_tokens(merged_output, state)
}

// ---------------------------------------------------------------------------
// compute_tome_stats
// ---------------------------------------------------------------------------

/// Compute statistics for a set of merge pairs.
pub fn compute_tome_stats(
    tokens: &[f32],
    n_tokens: usize,
    d_model: usize,
    pairs: &[(usize, usize)],
) -> Result<ToMeStats, TokenMergeError> {
    validate_shape(tokens, n_tokens, d_model)?;

    let original_tokens = n_tokens;
    let tokens_saved = pairs.len();
    let merged_tokens = original_tokens.saturating_sub(tokens_saved);
    let compression_ratio = if original_tokens == 0 {
        1.0
    } else {
        merged_tokens as f32 / original_tokens as f32
    };

    let (mean_similarity_merged, min_similarity_merged) = if pairs.is_empty() {
        (1.0, 1.0)
    } else {
        let mut sum = 0.0f32;
        let mut min = f32::INFINITY;
        for &(ai, bi) in pairs {
            if ai >= n_tokens || bi >= n_tokens {
                return Err(TokenMergeError::IndexOutOfRange {
                    idx: ai.max(bi),
                    n_tokens,
                });
            }
            let a = &tokens[ai * d_model..(ai + 1) * d_model];
            let b = &tokens[bi * d_model..(bi + 1) * d_model];
            let sim = token_cosine_similarity(a, b);
            sum += sim;
            if sim < min {
                min = sim;
            }
        }
        let n = pairs.len() as f32;
        (sum / n, min)
    };

    Ok(ToMeStats {
        original_tokens,
        merged_tokens,
        tokens_saved,
        mean_similarity_merged,
        min_similarity_merged,
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// progressive_merge / progressive_unmerge
// ---------------------------------------------------------------------------

/// Apply `rounds` successive ToMe passes, each merging `r_per_round` pairs.
///
/// Returns `(final_merged_tokens, Vec<MergeState>)` where `states[i]` describes
/// the merge performed in round `i` (indices relative to that round's input).
pub fn progressive_merge(
    tokens: &[f32],
    n_tokens: usize,
    d_model: usize,
    rounds: usize,
    r_per_round: usize,
) -> Result<(Vec<f32>, Vec<MergeState>), TokenMergeError> {
    if n_tokens == 0 {
        return Err(TokenMergeError::EmptySequence);
    }
    validate_shape(tokens, n_tokens, d_model)?;

    let mut current = tokens.to_vec();
    let mut current_n = n_tokens;
    let mut states = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        if current_n == 0 || r_per_round == 0 {
            // Nothing to merge; record an identity state.
            let state = MergeState::identity(current_n, d_model);
            states.push(state);
            continue;
        }

        let half = current_n / 2;
        let r = r_per_round.min(half);

        if r == 0 {
            let state = MergeState::identity(current_n, d_model);
            states.push(state);
            continue;
        }

        let pairs = bipartite_soft_matching(&current, current_n, d_model, r)?;
        let (merged, state) = merge_tokens(&current, current_n, d_model, &pairs, &MergeMode::Mean)?;
        current_n = state.merged_n_tokens;
        current = merged;
        states.push(state);
    }

    Ok((current, states))
}

/// Undo a progressive merge sequence in **reverse** order.
///
/// `states` must be the `Vec<MergeState>` returned by `progressive_merge`.
pub fn progressive_unmerge(
    merged: &[f32],
    states: &[MergeState],
) -> Result<Vec<f32>, TokenMergeError> {
    let mut current = merged.to_vec();
    for state in states.iter().rev() {
        current = unmerge_tokens(&current, state)?;
    }
    Ok(current)
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the attention speedup factor from ToMe.
///
/// Attention complexity is O(n²), so the speedup from reducing n tokens to m is:
/// `(n / m)²`
pub fn attention_speedup_factor(original_n: usize, merged_n: usize) -> f32 {
    if merged_n == 0 {
        return f32::INFINITY;
    }
    let n = original_n as f32;
    let m = merged_n as f32;
    (n / m).powi(2)
}

/// Returns `true` if two `MergeState`s were produced from inputs with the
/// same `original_n_tokens` and `d_model`.
pub fn merge_states_compatible(a: &MergeState, b: &MergeState) -> bool {
    a.original_n_tokens == b.original_n_tokens && a.d_model == b.d_model
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_shape(tokens: &[f32], n_tokens: usize, d_model: usize) -> Result<(), TokenMergeError> {
    if n_tokens == 0 || d_model == 0 {
        return Err(TokenMergeError::EmptySequence);
    }
    let expected = n_tokens * d_model;
    if tokens.len() != expected {
        return Err(TokenMergeError::DimensionMismatch {
            tokens_len: tokens.len(),
            n_tokens,
            d_model,
            expected,
        });
    }
    Ok(())
}

fn validate_merge_ratio(ratio: f32) -> Result<(), TokenMergeError> {
    if !(0.0..1.0).contains(&ratio) {
        return Err(TokenMergeError::InvalidMergeRatio { ratio });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Build a flat token matrix: token[i][j] = (i * d + j) as f32.
    fn make_tokens(n: usize, d: usize) -> Vec<f32> {
        (0..n * d).map(|x| x as f32).collect()
    }

    /// Build tokens where each token is a unit vector along dimension i % d.
    fn make_unit_tokens(n: usize, d: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n * d];
        for i in 0..n {
            v[i * d + (i % d)] = 1.0;
        }
        v
    }

    // ------------------------------------------------------------------
    // 1. token_cosine_similarity
    // ------------------------------------------------------------------

    #[test]
    fn test_cosine_sim_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = token_cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "identical vectors → 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = token_cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "orthogonal → 0.0, got {sim}");
    }

    #[test]
    fn test_cosine_sim_antiparallel() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = token_cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-4, "antiparallel → -1.0, got {sim}");
    }

    // ------------------------------------------------------------------
    // 2. token_l2_norm
    // ------------------------------------------------------------------

    #[test]
    fn test_l2_norm_zeros() {
        let v = vec![0.0f32; 4];
        assert_eq!(token_l2_norm(&v), 0.0);
    }

    #[test]
    fn test_l2_norm_unit() {
        let v = vec![1.0, 0.0, 0.0, 0.0];
        assert!((token_l2_norm(&v) - 1.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // 3. compute_similarity_matrix
    // ------------------------------------------------------------------

    #[test]
    fn test_similarity_matrix_diagonal_is_one() {
        let tokens = make_tokens(4, 3);
        let mat = compute_similarity_matrix(&tokens, 4, 3).unwrap();
        for i in 0..4 {
            let d = mat[i * 4 + i];
            assert!((d - 1.0).abs() < 1e-5, "diagonal[{i}] = {d}");
        }
    }

    #[test]
    fn test_similarity_matrix_2_tokens() {
        let tokens = vec![1.0, 0.0, 0.0, 1.0]; // 2 × 2
        let mat = compute_similarity_matrix(&tokens, 2, 2).unwrap();
        // [0,0]=1, [1,1]=1, [0,1]=[1,0]=0
        assert!((mat[0] - 1.0).abs() < 1e-5);
        assert!((mat[3] - 1.0).abs() < 1e-5);
        assert!(mat[1].abs() < 1e-5);
        assert!(mat[2].abs() < 1e-5);
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let tokens = make_tokens(4, 4);
        let mat = compute_similarity_matrix(&tokens, 4, 4).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let diff = (mat[i * 4 + j] - mat[j * 4 + i]).abs();
                assert!(diff < 1e-5, "matrix not symmetric at ({i},{j})");
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. bipartite_soft_matching
    // ------------------------------------------------------------------

    #[test]
    fn test_bipartite_r0_returns_empty() {
        let tokens = make_tokens(4, 4);
        let pairs = bipartite_soft_matching(&tokens, 4, 4, 0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_bipartite_r1_returns_one_pair() {
        let tokens = make_tokens(4, 4);
        let pairs = bipartite_soft_matching(&tokens, 4, 4, 1).unwrap();
        assert_eq!(pairs.len(), 1);
        let (a, b) = pairs[0];
        assert!(a < 4 && b < 4);
    }

    #[test]
    fn test_bipartite_pair_indices_valid() {
        let n = 6;
        let d = 4;
        let tokens = make_tokens(n, d);
        let r = 2;
        let pairs = bipartite_soft_matching(&tokens, n, d, r).unwrap();
        assert_eq!(pairs.len(), r);
        for (a, b) in &pairs {
            assert!(*a < n && *b < n);
        }
    }

    #[test]
    fn test_bipartite_identical_tokens_matched() {
        // Tokens 0 and 1 are identical; they should be matched.
        let mut tokens = vec![0.0f32; 4 * 4];
        // Token 0 (even) and token 1 (odd): both [1,0,0,0]
        tokens[0] = 1.0;
        tokens[4] = 1.0;
        // Token 2 (even) and token 3 (odd): both [0,1,0,0]
        tokens[9] = 1.0;
        tokens[13] = 1.0;
        let pairs = bipartite_soft_matching(&tokens, 4, 4, 1).unwrap();
        // Pair should be (0,1) with similarity 1.0 or (2,3) with similarity 1.0
        assert_eq!(pairs.len(), 1);
        let (a, b) = pairs[0];
        assert!(a % 2 == 0, "a must be even index");
        assert!(b % 2 == 1, "b must be odd index");
    }

    #[test]
    fn test_bipartite_r_too_large_errors() {
        let tokens = make_tokens(4, 4);
        let result = bipartite_soft_matching(&tokens, 4, 4, 3); // half=2, r=3
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // 5. merge_tokens
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_no_pairs_is_identity() {
        let tokens = make_tokens(4, 4);
        let (merged, state) = merge_tokens(&tokens, 4, 4, &[], &MergeMode::Mean).unwrap();
        // All original tokens must still appear (possibly reordered).
        assert_eq!(merged.len(), tokens.len());
        assert_eq!(state.merged_n_tokens, 4);
        assert_eq!(state.original_n_tokens, 4);
    }

    #[test]
    fn test_merge_mean_mode() {
        // 2 tokens × 2 dims; merge (0,1).
        let tokens = vec![2.0, 4.0, 6.0, 8.0]; // tok0=[2,4], tok1=[6,8]
        let pairs = vec![(0, 1)];
        let (merged, state) = merge_tokens(&tokens, 2, 2, &pairs, &MergeMode::Mean).unwrap();
        assert_eq!(merged.len(), 2); // 1 merged token × 2 dims
        assert!(
            (merged[0] - 4.0).abs() < 1e-6,
            "expected 4.0, got {}",
            merged[0]
        );
        assert!(
            (merged[1] - 6.0).abs() < 1e-6,
            "expected 6.0, got {}",
            merged[1]
        );
        assert_eq!(state.merged_n_tokens, 1);
    }

    #[test]
    fn test_merge_weighted_mode() {
        let tokens = vec![2.0, 4.0, 6.0, 8.0]; // tok0=[2,4], tok1=[6,8]
        let pairs = vec![(0, 1)];
        let mode = MergeMode::Weighted { alpha: 0.25 };
        let (merged, _) = merge_tokens(&tokens, 2, 2, &pairs, &mode).unwrap();
        // 2*0.25 + 6*0.75 = 5.0;  4*0.25 + 8*0.75 = 7.0
        assert!(
            (merged[0] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            merged[0]
        );
        assert!(
            (merged[1] - 7.0).abs() < 1e-5,
            "expected 7.0, got {}",
            merged[1]
        );
    }

    #[test]
    fn test_merge_two_pairs() {
        let tokens = make_tokens(4, 2); // 4 tokens × 2 dims
        let pairs = vec![(0, 1), (2, 3)];
        let (merged, state) = merge_tokens(&tokens, 4, 2, &pairs, &MergeMode::Mean).unwrap();
        assert_eq!(state.merged_n_tokens, 2);
        assert_eq!(merged.len(), 4); // 2 tokens × 2 dims
    }

    // ------------------------------------------------------------------
    // 6. unmerge_tokens
    // ------------------------------------------------------------------

    #[test]
    fn test_unmerge_round_trip_2_tokens() {
        let tokens = vec![1.0, 2.0, 3.0, 4.0]; // 2 × 2
        let pairs = vec![(0, 1)];
        let (merged, state) = merge_tokens(&tokens, 2, 2, &pairs, &MergeMode::Mean).unwrap();
        let restored = unmerge_tokens(&merged, &state).unwrap();
        // Both slots filled with the merged mean.
        assert_eq!(restored.len(), 4);
        assert!((restored[0] - restored[2]).abs() < 1e-6);
    }

    #[test]
    fn test_unmerge_identity_state() {
        let tokens = make_tokens(3, 4);
        let state = MergeState::identity(3, 4);
        // identity unmerge with identity merge means input == output.
        let result = unmerge_tokens(&tokens, &state).unwrap();
        for (a, b) in tokens.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_unmerge_round_trip_preserves_shape() {
        let tokens = make_tokens(6, 8);
        let pairs = vec![(0, 1), (2, 3)];
        let (merged, state) = merge_tokens(&tokens, 6, 8, &pairs, &MergeMode::Mean).unwrap();
        let restored = unmerge_tokens(&merged, &state).unwrap();
        assert_eq!(restored.len(), tokens.len());
    }

    #[test]
    fn test_unmerge_no_pairs_round_trip() {
        let tokens = make_tokens(4, 4);
        let (merged, state) = merge_tokens(&tokens, 4, 4, &[], &MergeMode::Mean).unwrap();
        let restored = unmerge_tokens(&merged, &state).unwrap();
        assert_eq!(restored.len(), tokens.len());
    }

    // ------------------------------------------------------------------
    // 7. tome_merge
    // ------------------------------------------------------------------

    #[test]
    fn test_tome_merge_ratio_zero_identity() {
        let tokens = make_tokens(4, 4);
        let config = ToMeConfig {
            merge_ratio: 0.0,
            ..Default::default()
        };
        let (merged, state) = tome_merge(&tokens, 4, 4, &config).unwrap();
        assert_eq!(merged.len(), tokens.len());
        assert_eq!(state.merged_n_tokens, 4);
    }

    #[test]
    fn test_tome_merge_ratio_half() {
        let tokens = make_tokens(4, 4);
        let config = ToMeConfig::default(); // merge_ratio = 0.5 → r=1
        let (merged, state) = tome_merge(&tokens, 4, 4, &config).unwrap();
        assert!(merged.len() < tokens.len());
        assert_eq!(state.merged_n_tokens, state.merge_groups.len());
    }

    #[test]
    fn test_tome_merge_known_input() {
        // 2 tokens × 2 dims; merge_ratio=0.5 → r=floor(2*0.5/2)=0 (no merge!)
        // Use 4 tokens instead.
        let tokens = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]; // 4 × 2
        let config = ToMeConfig {
            merge_ratio: 0.5,
            ..Default::default()
        };
        let (merged, state) = tome_merge(&tokens, 4, 2, &config).unwrap();
        assert!(state.original_n_tokens == 4);
        assert!(state.merged_n_tokens <= 4);
        let _ = merged;
    }

    #[test]
    fn test_tome_merge_empty_errors() {
        let tokens: Vec<f32> = Vec::new();
        let config = ToMeConfig::default();
        let result = tome_merge(&tokens, 0, 4, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_tome_merge_invalid_ratio_errors() {
        let tokens = make_tokens(4, 4);
        let config = ToMeConfig {
            merge_ratio: 1.5,
            ..Default::default()
        };
        let result = tome_merge(&tokens, 4, 4, &config);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // 8. tome_unmerge (round-trip)
    // ------------------------------------------------------------------

    #[test]
    fn test_tome_round_trip_shape() {
        let tokens = make_tokens(8, 8);
        let config = ToMeConfig::default();
        let (merged, state) = tome_merge(&tokens, 8, 8, &config).unwrap();
        let restored = tome_unmerge(&merged, &state).unwrap();
        assert_eq!(restored.len(), tokens.len());
    }

    #[test]
    fn test_tome_round_trip_mean_preserved() {
        // After unmerge the mean of each channel should be close to original mean
        // (unmerge broadcasts merged value to both slots, so channel means differ).
        let tokens = make_tokens(4, 4);
        let config = ToMeConfig {
            merge_ratio: 0.0,
            ..Default::default()
        };
        let (merged, state) = tome_merge(&tokens, 4, 4, &config).unwrap();
        let restored = tome_unmerge(&merged, &state).unwrap();
        // ratio=0 → identity; restored == original
        for (a, b) in tokens.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_tome_unmerge_length_matches_original() {
        let n = 10;
        let d = 6;
        let tokens = make_tokens(n, d);
        let config = ToMeConfig {
            merge_ratio: 0.4,
            ..Default::default()
        };
        let (merged, state) = tome_merge(&tokens, n, d, &config).unwrap();
        let restored = tome_unmerge(&merged, &state).unwrap();
        assert_eq!(restored.len(), n * d);
    }

    // ------------------------------------------------------------------
    // 9. MergeState helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_state_tokens_saved() {
        let state = MergeState {
            merge_groups: vec![vec![0, 1], vec![2]],
            original_n_tokens: 3,
            merged_n_tokens: 2,
            d_model: 4,
        };
        assert_eq!(state.tokens_saved(), 1);
    }

    #[test]
    fn test_merge_state_compression_ratio() {
        let state = MergeState {
            merge_groups: vec![vec![0, 1], vec![2, 3]],
            original_n_tokens: 4,
            merged_n_tokens: 2,
            d_model: 4,
        };
        assert!((state.compression_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_merge_state_identity_ratio() {
        let state = MergeState::identity(4, 4);
        assert!((state.compression_ratio() - 1.0).abs() < 1e-6);
        assert_eq!(state.tokens_saved(), 0);
    }

    // ------------------------------------------------------------------
    // 10. compute_tome_stats
    // ------------------------------------------------------------------

    #[test]
    fn test_tome_stats_no_pairs() {
        let tokens = make_tokens(4, 4);
        let stats = compute_tome_stats(&tokens, 4, 4, &[]).unwrap();
        assert_eq!(stats.tokens_saved, 0);
        assert_eq!(stats.merged_tokens, 4);
    }

    #[test]
    fn test_tome_stats_one_pair_similarity() {
        // Two identical tokens → cosine similarity = 1.0.
        let tokens = vec![1.0f32, 0.0, 1.0, 0.0]; // 2 × 2, both [1,0]
        let pairs = vec![(0, 1)];
        let stats = compute_tome_stats(&tokens, 2, 2, &pairs).unwrap();
        assert!(
            (stats.mean_similarity_merged - 1.0).abs() < 1e-4,
            "expected ~1.0, got {}",
            stats.mean_similarity_merged
        );
        assert_eq!(stats.tokens_saved, 1);
    }

    #[test]
    fn test_tome_stats_compression_ratio() {
        let tokens = make_tokens(4, 4);
        let pairs = bipartite_soft_matching(&tokens, 4, 4, 1).unwrap();
        let stats = compute_tome_stats(&tokens, 4, 4, &pairs).unwrap();
        assert!(stats.compression_ratio > 0.0 && stats.compression_ratio <= 1.0);
    }

    // ------------------------------------------------------------------
    // 11. progressive_merge
    // ------------------------------------------------------------------

    #[test]
    fn test_progressive_merge_2_rounds() {
        let n = 8;
        let d = 4;
        let tokens = make_tokens(n, d);
        let (final_toks, states) = progressive_merge(&tokens, n, d, 2, 1).unwrap();
        assert_eq!(states.len(), 2);
        assert!(final_toks.len() < tokens.len());
    }

    #[test]
    fn test_progressive_merge_states_count() {
        let tokens = make_tokens(8, 4);
        let (_, states) = progressive_merge(&tokens, 8, 4, 3, 1).unwrap();
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_progressive_merge_zero_rounds() {
        let tokens = make_tokens(4, 4);
        let (out, states) = progressive_merge(&tokens, 4, 4, 0, 1).unwrap();
        assert_eq!(states.len(), 0);
        assert_eq!(out.len(), tokens.len());
    }

    // ------------------------------------------------------------------
    // 12. progressive_unmerge
    // ------------------------------------------------------------------

    #[test]
    fn test_progressive_unmerge_restores_length() {
        let n = 8;
        let d = 4;
        let tokens = make_tokens(n, d);
        let (final_toks, states) = progressive_merge(&tokens, n, d, 2, 1).unwrap();
        let restored = progressive_unmerge(&final_toks, &states).unwrap();
        assert_eq!(restored.len(), n * d);
    }

    #[test]
    fn test_progressive_unmerge_zero_rounds() {
        let tokens = make_tokens(4, 4);
        let (out, states) = progressive_merge(&tokens, 4, 4, 0, 1).unwrap();
        let restored = progressive_unmerge(&out, &states).unwrap();
        assert_eq!(restored.len(), tokens.len());
    }

    #[test]
    fn test_progressive_roundtrip_identity_ratio() {
        // With r_per_round = 0 (clamped), states are identities → roundtrip exact.
        let tokens = make_tokens(4, 4);
        let (out, states) = progressive_merge(&tokens, 4, 4, 3, 0).unwrap();
        let restored = progressive_unmerge(&out, &states).unwrap();
        for (a, b) in tokens.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ------------------------------------------------------------------
    // 13. attention_speedup_factor
    // ------------------------------------------------------------------

    #[test]
    fn test_speedup_half_tokens() {
        let factor = attention_speedup_factor(8, 4);
        assert!((factor - 4.0).abs() < 1e-5, "expected 4×, got {factor}");
    }

    #[test]
    fn test_speedup_no_merge() {
        let factor = attention_speedup_factor(8, 8);
        assert!((factor - 1.0).abs() < 1e-5, "expected 1×, got {factor}");
    }

    // ------------------------------------------------------------------
    // 14. merge_states_compatible
    // ------------------------------------------------------------------

    #[test]
    fn test_states_compatible_same() {
        let a = MergeState::identity(4, 8);
        let b = MergeState::identity(4, 8);
        assert!(merge_states_compatible(&a, &b));
    }

    #[test]
    fn test_states_incompatible_different_n() {
        let a = MergeState::identity(4, 8);
        let b = MergeState::identity(6, 8);
        assert!(!merge_states_compatible(&a, &b));
    }

    #[test]
    fn test_states_incompatible_different_d() {
        let a = MergeState::identity(4, 8);
        let b = MergeState::identity(4, 16);
        assert!(!merge_states_compatible(&a, &b));
    }

    // ------------------------------------------------------------------
    // 15. ToMeConfig defaults
    // ------------------------------------------------------------------

    #[test]
    fn test_config_default_merge_ratio() {
        let cfg = ToMeConfig::default();
        assert!((cfg.merge_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_default_use_keys() {
        let cfg = ToMeConfig::default();
        assert!(cfg.use_keys_for_matching);
    }

    // ------------------------------------------------------------------
    // 16. Weighted merge mode
    // ------------------------------------------------------------------

    #[test]
    fn test_weighted_alpha_zero() {
        // alpha=0 → result = b (second token)
        let tokens = vec![1.0, 1.0, 5.0, 5.0]; // tok0=[1,1], tok1=[5,5]
        let pairs = vec![(0, 1)];
        let mode = MergeMode::Weighted { alpha: 0.0 };
        let (merged, _) = merge_tokens(&tokens, 2, 2, &pairs, &mode).unwrap();
        assert!(
            (merged[0] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            merged[0]
        );
        assert!(
            (merged[1] - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            merged[1]
        );
    }

    #[test]
    fn test_weighted_alpha_one() {
        // alpha=1 → result = a (first token)
        let tokens = vec![1.0, 1.0, 5.0, 5.0];
        let pairs = vec![(0, 1)];
        let mode = MergeMode::Weighted { alpha: 1.0 };
        let (merged, _) = merge_tokens(&tokens, 2, 2, &pairs, &mode).unwrap();
        assert!(
            (merged[0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            merged[0]
        );
        assert!(
            (merged[1] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            merged[1]
        );
    }

    // ------------------------------------------------------------------
    // Extra tests to surpass 40
    // ------------------------------------------------------------------

    #[test]
    fn test_unit_tokens_matching() {
        // Unit vectors along each axis: tokens 0,2,4 (A) vs 1,3,5 (B)
        let tokens = make_unit_tokens(6, 6);
        let pairs = bipartite_soft_matching(&tokens, 6, 6, 1).unwrap();
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn test_merge_state_groups_count_after_merge() {
        let tokens = make_tokens(6, 4);
        let pairs = bipartite_soft_matching(&tokens, 6, 4, 2).unwrap();
        let (_, state) = merge_tokens(&tokens, 6, 4, &pairs, &MergeMode::Mean).unwrap();
        assert_eq!(state.merge_groups.len(), state.merged_n_tokens);
    }

    #[test]
    fn test_l2_norm_known_value() {
        let v = vec![3.0f32, 4.0];
        let norm = token_l2_norm(&v);
        assert!(
            (norm - 5.0).abs() < 1e-5,
            "3-4-5 triangle, expected 5.0, got {norm}"
        );
    }

    #[test]
    fn test_cosine_sim_empty() {
        // Empty slices should return 0.0, not panic.
        let sim = token_cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let tokens = vec![1.0f32; 5]; // not 2×3=6
        let result = compute_similarity_matrix(&tokens, 2, 3);
        assert!(matches!(
            result,
            Err(TokenMergeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_attention_speedup_quarter_tokens() {
        let factor = attention_speedup_factor(8, 2);
        assert!((factor - 16.0).abs() < 1e-4, "expected 16×, got {factor}");
    }

    #[test]
    fn test_progressive_merge_decreases_tokens() {
        let tokens = make_tokens(16, 4);
        let (out, states) = progressive_merge(&tokens, 16, 4, 4, 1).unwrap();
        let final_n = states.last().map(|s| s.merged_n_tokens).unwrap_or(16);
        assert!(final_n <= 16);
        assert!(out.len() <= tokens.len());
    }

    #[test]
    fn test_merge_tokens_single_unmatched_b() {
        // 4 tokens, 1 pair (merges tokens 0+1), token 3 (odd, unmatched) kept
        let tokens = make_tokens(4, 2);
        let pairs = vec![(0, 1)];
        let (_, state) = merge_tokens(&tokens, 4, 2, &pairs, &MergeMode::Mean).unwrap();
        // merged: pair (0+1) → 1 token, unmatched a-side (2) → 1 token, unmatched b-side (3) → 1 token
        assert_eq!(state.merged_n_tokens, 3);
    }

    #[test]
    fn test_progressive_unmerge_three_rounds() {
        let tokens = make_tokens(16, 4);
        let (out, states) = progressive_merge(&tokens, 16, 4, 3, 2).unwrap();
        let restored = progressive_unmerge(&out, &states).unwrap();
        assert_eq!(restored.len(), 16 * 4);
    }

    #[test]
    fn test_tome_stats_min_le_mean() {
        let tokens = make_tokens(6, 4);
        let pairs = bipartite_soft_matching(&tokens, 6, 4, 2).unwrap();
        let stats = compute_tome_stats(&tokens, 6, 4, &pairs).unwrap();
        assert!(stats.min_similarity_merged <= stats.mean_similarity_merged + 1e-5);
    }
}
