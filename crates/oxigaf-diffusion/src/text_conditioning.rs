//! Text prompt conditioning for diffusion models.
//!
//! Provides a whitespace/punctuation tokenizer with a fixed vocabulary and an
//! embedding look-up table that turns prompts such as
//! `"frontal portrait, neutral expression"` into dense `[max_length × embed_dim]`
//! buffers suitable for cross-attention.
//!
//! # Real weights vs. placeholder weights
//!
//! The embedding table is the only part that needs training:
//!
//! - [`TextConditioner::from_pretrained`] loads a real embedding matrix from a
//!   safetensors file plus a vocabulary file. Use this whenever prompts are
//!   expected to influence generation.
//! - [`TextConditioner::new`] falls back to a **random placeholder** table.
//!   Random vectors carry no semantic information: different prompts produce
//!   different *noise*, not different meanings. The placeholder exists so the
//!   shape/plumbing of the conditioning path can be exercised without weights;
//!   it is flagged by [`TextConditioner::is_placeholder`] and logs a warning
//!   when constructed.
//!
//! Nothing here is a CLIP text encoder: the look-up table is a per-token
//! embedding, so a pre-trained *token embedding* matrix (for example the input
//! embedding of a CLIP text model) is what
//! [`TextConditioner::from_pretrained`] expects.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device};

use crate::{DiffusionError, DiffusionResult};

// ---------------------------------------------------------------------------
// SimpleTokenizer
// ---------------------------------------------------------------------------

/// Simple whitespace-and-punctuation tokenizer with a fixed vocabulary.
///
/// Special token IDs:
/// - 0 = PAD
/// - 1 = UNK
/// - 2 = BOS (beginning of sequence)
/// - 3 = EOS (end of sequence)
/// - 4..= vocab words in insertion order
pub struct SimpleTokenizer {
    vocab: HashMap<String, u32>,
    /// token_id → word (index 0-3 are the special-token placeholders).
    inverse_vocab: Vec<String>,
    pub pad_token_id: u32,
    pub unk_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    /// Sequence length produced by [`Self::encode_padded`]; also the truncation
    /// limit of [`Self::tokenize`].
    pub max_length: usize,
}

impl SimpleTokenizer {
    /// Build a tokenizer from a slice of vocabulary words.
    ///
    /// IDs are assigned: 0=PAD, 1=UNK, 2=BOS, 3=EOS, then vocab words starting
    /// at 4.  Words are lower-cased before insertion.
    ///
    /// `max_length` defaults to 77 (the CLIP context length); use
    /// [`Self::with_max_length`] to pick another one.
    pub fn new(vocab_words: &[&str]) -> Self {
        Self::with_max_length(vocab_words, 77)
    }

    /// Build a tokenizer with an explicit sequence length.
    pub fn with_max_length(vocab_words: &[&str], max_length: usize) -> Self {
        let pad_token_id: u32 = 0;
        let unk_token_id: u32 = 1;
        let bos_token_id: u32 = 2;
        let eos_token_id: u32 = 3;

        let mut inverse_vocab: Vec<String> = vec![
            "<PAD>".to_string(),
            "<UNK>".to_string(),
            "<BOS>".to_string(),
            "<EOS>".to_string(),
        ];
        let mut vocab: HashMap<String, u32> = HashMap::new();

        for word in vocab_words {
            let lower = word.to_lowercase();
            if !vocab.contains_key(&lower) {
                let id = inverse_vocab.len() as u32;
                vocab.insert(lower.clone(), id);
                inverse_vocab.push(lower);
            }
        }

        Self {
            vocab,
            inverse_vocab,
            pad_token_id,
            unk_token_id,
            bos_token_id,
            eos_token_id,
            max_length,
        }
    }

    /// Tokenizer pre-loaded with common face / avatar vocabulary.
    pub fn default_face_vocab() -> Self {
        let words = &[
            "face",
            "portrait",
            "head",
            "expression",
            "neutral",
            "smile",
            "frown",
            "happy",
            "sad",
            "angry",
            "surprised",
            "fear",
            "disgust",
            "frontal",
            "profile",
            "left",
            "right",
            "three",
            "quarter",
            "view",
            "young",
            "old",
            "man",
            "woman",
            "person",
            "avatar",
            "render",
            "3d",
            "gaussian",
            "realistic",
            "cartoon",
            "artistic",
            "high",
            "quality",
        ];
        Self::new(words)
    }

    /// Load a vocabulary from a text file, one token per line.
    ///
    /// Blank lines and lines starting with `#` are ignored; tokens are
    /// lower-cased. The four special tokens are prepended automatically, so the
    /// file must contain only real words.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::ModelLoad`] when the file cannot be read.
    /// - [`DiffusionError::InvalidConfig`] when it contains no usable token.
    pub fn from_vocab_file(path: &Path, max_length: usize) -> DiffusionResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            DiffusionError::ModelLoad(format!(
                "Failed to read vocabulary file {}: {e}",
                path.display()
            ))
        })?;

        let words: Vec<String> = text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| line.to_lowercase())
            .collect();

        if words.is_empty() {
            return Err(DiffusionError::InvalidConfig(format!(
                "vocabulary file {} contains no tokens",
                path.display()
            )));
        }

        let refs: Vec<&str> = words.iter().map(|w| w.as_str()).collect();
        Ok(Self::with_max_length(&refs, max_length))
    }

    /// Split `text` into lowercase alphabetical/numeric tokens (splitting on
    /// any non-alphanumeric character including whitespace and punctuation).
    fn split_into_words(text: &str) -> Vec<String> {
        let mut words: Vec<String> = Vec::new();
        let mut current: String = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() {
                current.push(ch);
            } else if !current.is_empty() {
                words.push(current.to_lowercase());
                current = String::new();
            }
        }
        if !current.is_empty() {
            words.push(current.to_lowercase());
        }
        words
    }

    /// Tokenize `text`.
    ///
    /// Returns `[BOS, w1, w2, …, EOS]`, never longer than `max_length`.
    /// Words beyond the budget are dropped **before** the EOS marker is
    /// appended, so a truncated prompt still terminates with EOS.
    /// Unknown words are mapped to `unk_token_id`.
    ///
    /// Degenerate lengths: `max_length == 0` yields an empty sequence and
    /// `max_length == 1` yields `[BOS]`.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        match self.max_length {
            0 => return Vec::new(),
            1 => return vec![self.bos_token_id],
            _ => {}
        }

        // Reserve room for BOS and EOS.
        let word_budget = self.max_length - 2;
        let words = Self::split_into_words(text);

        let mut tokens: Vec<u32> = Vec::with_capacity(words.len().min(word_budget) + 2);
        tokens.push(self.bos_token_id);
        for word in words.iter().take(word_budget) {
            let id = self.vocab.get(word).copied().unwrap_or(self.unk_token_id);
            tokens.push(id);
        }
        tokens.push(self.eos_token_id);
        tokens
    }

    /// Tokenize and right-pad with `pad_token_id` to exactly `max_length`.
    pub fn encode_padded(&self, text: &str) -> Vec<u32> {
        let mut tokens = self.tokenize(text);
        tokens.resize(self.max_length, self.pad_token_id);
        tokens
    }

    /// Map a token sequence back to a string, skipping PAD / BOS / EOS tokens.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let special = [self.pad_token_id, self.bos_token_id, self.eos_token_id];
        tokens
            .iter()
            .filter(|&&id| !special.contains(&id))
            .filter_map(|&id| self.inverse_vocab.get(id as usize).map(|s| s.as_str()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Total size of the vocabulary (including special tokens).
    pub fn vocab_size(&self) -> usize {
        self.inverse_vocab.len()
    }
}

// ---------------------------------------------------------------------------
// TextEmbedding
// ---------------------------------------------------------------------------

/// Token embedding look-up table.
///
/// Weights are stored as a flat `[vocab_size × embed_dim]` `f32` buffer;
/// `lookup(id)` returns a `&[f32]` slice of length `embed_dim`.
///
/// Tables built by [`Self::new`] or [`Self::zeros`] are *placeholders* — see
/// [`Self::is_placeholder`]. Real weights come from [`Self::from_safetensors`]
/// or [`Self::from_weights`].
pub struct TextEmbedding {
    pub vocab_size: usize,
    pub embed_dim: usize,
    /// Flat weight buffer: `weights[id * embed_dim..(id+1) * embed_dim]`.
    weights: Vec<f32>,
    /// Zero row returned for out-of-bounds token IDs.
    zero_row: Vec<f32>,
    /// `true` when the weights are synthetic rather than trained.
    placeholder: bool,
}

impl TextEmbedding {
    /// Create a **placeholder** embedding table with xorshift64 random
    /// initialisation.
    ///
    /// Values are drawn uniformly from `[-0.02, 0.02]`.
    /// If `seed` is 0 the guard value `0xDEAD_BEEF` is used instead.
    ///
    /// The result is deterministic but semantically meaningless: it is only
    /// useful for shape/plumbing tests. Load real weights with
    /// [`Self::from_safetensors`] for prompts that must influence generation.
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        let mut state: u64 = if seed == 0 { 0xDEAD_BEEF } else { seed };
        let total = vocab_size * embed_dim;
        let mut weights = Vec::with_capacity(total);

        for _ in 0..total {
            // xorshift64 step
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Convert to float in [0, 1)
            let rand_val = (state >> 11) as f64 / (1u64 << 53) as f64;
            // Map to [-0.02, 0.02]
            let value = (2.0 * rand_val - 1.0) * 0.02;
            weights.push(value as f32);
        }

        let zero_row = vec![0.0_f32; embed_dim];
        Self {
            vocab_size,
            embed_dim,
            weights,
            zero_row,
            placeholder: true,
        }
    }

    /// Create an all-zeros **placeholder** embedding table.
    pub fn zeros(vocab_size: usize, embed_dim: usize) -> Self {
        let total = vocab_size * embed_dim;
        let zero_row = vec![0.0_f32; embed_dim];
        Self {
            vocab_size,
            embed_dim,
            weights: vec![0.0_f32; total],
            zero_row,
            placeholder: true,
        }
    }

    /// Build a table from an existing `[vocab_size × embed_dim]` weight buffer.
    ///
    /// The result is *not* flagged as a placeholder.
    ///
    /// # Errors
    ///
    /// [`DiffusionError::InvalidConfig`] when the buffer length does not match
    /// `vocab_size * embed_dim`, or when either dimension is zero.
    pub fn from_weights(
        vocab_size: usize,
        embed_dim: usize,
        weights: Vec<f32>,
    ) -> DiffusionResult<Self> {
        if vocab_size == 0 || embed_dim == 0 {
            return Err(DiffusionError::InvalidConfig(
                "vocab_size and embed_dim must be > 0".to_string(),
            ));
        }
        let expected = vocab_size * embed_dim;
        if weights.len() != expected {
            return Err(DiffusionError::InvalidConfig(format!(
                "embedding weights have {} elements but expected {expected} ({vocab_size} × {embed_dim})",
                weights.len()
            )));
        }
        let zero_row = vec![0.0_f32; embed_dim];
        Ok(Self {
            vocab_size,
            embed_dim,
            weights,
            zero_row,
            placeholder: false,
        })
    }

    /// Load a trained token-embedding matrix from a safetensors file.
    ///
    /// `tensor_name` must reference a 2-D `[vocab_size, embed_dim]` tensor (for
    /// example `text_model.embeddings.token_embedding.weight` in a CLIP text
    /// encoder checkpoint). Any float dtype is converted to `f32`.
    ///
    /// # Errors
    ///
    /// [`DiffusionError::ModelLoad`] when the file cannot be read, the tensor
    /// is missing, is not 2-D, or cannot be converted to `f32`.
    pub fn from_safetensors(path: &Path, tensor_name: &str) -> DiffusionResult<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu).map_err(|e| {
            DiffusionError::ModelLoad(format!(
                "Failed to load text embedding weights from {}: {e}",
                path.display()
            ))
        })?;

        let tensor = tensors.get(tensor_name).ok_or_else(|| {
            let mut available: Vec<&str> = tensors.keys().map(|k| k.as_str()).collect();
            available.sort_unstable();
            DiffusionError::ModelLoad(format!(
                "Text embedding tensor '{tensor_name}' not found in {} (available: {})",
                path.display(),
                available.join(", ")
            ))
        })?;

        let (vocab_size, embed_dim) = tensor.dims2().map_err(|e| {
            DiffusionError::ModelLoad(format!(
                "Text embedding tensor '{tensor_name}' must be 2-D [vocab_size, embed_dim]: {e}"
            ))
        })?;

        let weights = tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                DiffusionError::ModelLoad(format!(
                    "Failed to read text embedding tensor '{tensor_name}' as f32: {e}"
                ))
            })?;

        Self::from_weights(vocab_size, embed_dim, weights)
    }

    /// `true` when the weights are synthetic (random or zero) rather than
    /// trained, i.e. prompts cannot carry meaning through this table.
    pub fn is_placeholder(&self) -> bool {
        self.placeholder
    }

    /// Return the embedding vector for `token_id`.
    ///
    /// Returns a slice of `embed_dim` zeros if `token_id` is out of bounds.
    pub fn lookup(&self, token_id: u32) -> &[f32] {
        let id = token_id as usize;
        if id >= self.vocab_size {
            return &self.zero_row;
        }
        let start = id * self.embed_dim;
        let end = start + self.embed_dim;
        // Bounds are guaranteed by the constructors (weights.len() ==
        // vocab_size * embed_dim), but fall back to the zero row rather than
        // risking a slice panic if that invariant is ever violated.
        self.weights.get(start..end).unwrap_or(&self.zero_row)
    }

    /// Embed a token sequence.
    ///
    /// Returns a flat buffer of shape `[seq_len × embed_dim]` by concatenating
    /// the embedding vector for each token.
    pub fn embed_sequence(&self, tokens: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(tokens.len() * self.embed_dim);
        for &tok in tokens {
            out.extend_from_slice(self.lookup(tok));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// TextConditioningConfig
// ---------------------------------------------------------------------------

/// Configuration for text conditioning.
#[derive(Debug, Clone)]
pub struct TextConditioningConfig {
    /// Embedding dimensionality (CLIP default: 768).
    pub embed_dim: usize,
    /// Maximum token sequence length (CLIP default: 77).
    pub max_length: usize,
    /// Classifier-free guidance scale.
    pub guidance_scale: f32,
    /// Probability of dropping the conditioning during training (CFG).
    pub dropout_prob: f32,
    /// Whether to use a negative prompt for unconditional embedding.
    pub use_negative_prompt: bool,
    /// Default negative prompt text.
    pub negative_prompt: String,
}

impl Default for TextConditioningConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            max_length: 77,
            guidance_scale: 7.5,
            dropout_prob: 0.1,
            use_negative_prompt: false,
            negative_prompt: "blurry, low quality, distorted".to_string(),
        }
    }
}

impl TextConditioningConfig {
    /// Create a configuration with a custom guidance scale, all other fields at
    /// their defaults.
    pub fn with_guidance(scale: f32) -> Self {
        Self {
            guidance_scale: scale,
            ..Self::default()
        }
    }

    /// Validate that all parameters are logically consistent.
    pub fn validate(&self) -> DiffusionResult<()> {
        if self.embed_dim == 0 {
            return Err(DiffusionError::InvalidConfig(
                "embed_dim must be > 0".to_string(),
            ));
        }
        if self.max_length == 0 {
            return Err(DiffusionError::InvalidConfig(
                "max_length must be > 0".to_string(),
            ));
        }
        if self.guidance_scale <= 0.0 {
            return Err(DiffusionError::InvalidConfig(
                "guidance_scale must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TextConditioner
// ---------------------------------------------------------------------------

/// Seed used for the placeholder embedding table.
const PLACEHOLDER_EMBEDDING_SEED: u64 = 42;

/// End-to-end text conditioning pipeline for a diffusion model.
///
/// Combines a [`SimpleTokenizer`] and a [`TextEmbedding`] table to convert
/// raw text strings into dense embeddings suitable for cross-attention.
///
/// The tokenizer's `max_length` always mirrors `config.max_length`, so every
/// buffer this type returns has exactly `config.max_length * config.embed_dim`
/// elements.
pub struct TextConditioner {
    pub tokenizer: SimpleTokenizer,
    pub embedding: TextEmbedding,
    pub config: TextConditioningConfig,
}

impl TextConditioner {
    /// Construct a conditioner backed by a **placeholder** embedding table.
    ///
    /// Uses the default face vocabulary and a deterministic random table.
    /// Prompts will *not* carry meaning through such a table — see the module
    /// documentation — so a warning is logged. Use [`Self::from_pretrained`]
    /// for real weights.
    pub fn new(config: TextConditioningConfig) -> Self {
        let mut tokenizer = SimpleTokenizer::default_face_vocab();
        tokenizer.max_length = config.max_length;
        let vocab_size = tokenizer.vocab_size();
        let embedding =
            TextEmbedding::new(vocab_size, config.embed_dim, PLACEHOLDER_EMBEDDING_SEED);

        tracing::warn!(
            "TextConditioner built with placeholder embeddings ({} tokens × {} dims): \
             prompts carry no semantic signal. Use TextConditioner::from_pretrained \
             to load trained weights.",
            vocab_size,
            config.embed_dim
        );

        Self {
            tokenizer,
            embedding,
            config,
        }
    }

    /// Like [`Self::new`], but validates `config` first.
    ///
    /// # Errors
    ///
    /// Anything [`TextConditioningConfig::validate`] reports.
    pub fn try_new(config: TextConditioningConfig) -> DiffusionResult<Self> {
        config.validate()?;
        Ok(Self::new(config))
    }

    /// Build a conditioner from trained weights.
    ///
    /// - `embedding_path`: safetensors file holding the token-embedding matrix.
    /// - `tensor_name`: name of the `[vocab_size, embed_dim]` tensor inside it.
    /// - `vocab_path`: text file with one vocabulary token per line, in the
    ///   order matching the embedding rows (offset by the four special tokens).
    ///
    /// # Errors
    ///
    /// - Anything [`TextConditioningConfig::validate`],
    ///   [`SimpleTokenizer::from_vocab_file`] or
    ///   [`TextEmbedding::from_safetensors`] reports.
    /// - [`DiffusionError::InvalidConfig`] when the loaded table does not match
    ///   `config.embed_dim` or is too small for the vocabulary.
    pub fn from_pretrained(
        config: TextConditioningConfig,
        embedding_path: &Path,
        tensor_name: &str,
        vocab_path: &Path,
    ) -> DiffusionResult<Self> {
        config.validate()?;

        let tokenizer = SimpleTokenizer::from_vocab_file(vocab_path, config.max_length)?;
        let embedding = TextEmbedding::from_safetensors(embedding_path, tensor_name)?;

        if embedding.embed_dim != config.embed_dim {
            return Err(DiffusionError::InvalidConfig(format!(
                "embedding dimension mismatch: weights provide {} but config requests {}",
                embedding.embed_dim, config.embed_dim
            )));
        }
        if embedding.vocab_size < tokenizer.vocab_size() {
            return Err(DiffusionError::InvalidConfig(format!(
                "embedding table holds {} rows but the vocabulary needs {}",
                embedding.vocab_size,
                tokenizer.vocab_size()
            )));
        }

        Ok(Self {
            tokenizer,
            embedding,
            config,
        })
    }

    /// `true` when the embedding table is synthetic (see [`Self::new`]).
    pub fn is_placeholder(&self) -> bool {
        self.embedding.is_placeholder()
    }

    /// Sequence length of every buffer this conditioner returns.
    pub fn sequence_length(&self) -> usize {
        self.tokenizer.max_length
    }

    /// Number of `f32` elements in one encoded prompt
    /// (`sequence_length() * embed_dim`).
    pub fn embedding_len(&self) -> usize {
        self.sequence_length() * self.embedding.embed_dim
    }

    /// Encode a single text prompt.
    ///
    /// Returns a flat buffer of shape `[max_length × embed_dim]`, where
    /// `max_length` is `config.max_length`.
    pub fn encode_text(&self, text: &str) -> Vec<f32> {
        let tokens = self.tokenizer.encode_padded(text);
        self.embedding.embed_sequence(&tokens)
    }

    /// Encode text for classifier-free guidance.
    ///
    /// Returns `(conditional_embedding, unconditional_embedding)`; both buffers
    /// always have the same length.
    /// The unconditional embedding is either the encoding of
    /// `config.negative_prompt` (when `use_negative_prompt` is true) or an
    /// all-zeros buffer of the same shape.
    pub fn encode_with_cfg(&self, text: &str) -> (Vec<f32>, Vec<f32>) {
        let cond = self.encode_text(text);
        let uncond = if self.config.use_negative_prompt {
            self.encode_text(&self.config.negative_prompt.clone())
        } else {
            // Mirror the conditional length exactly: `config.max_length` and the
            // tokenizer length are kept in sync, but deriving the length from
            // `cond` keeps the two halves consistent even if a caller mutates
            // `tokenizer.max_length` directly.
            vec![0.0_f32; cond.len()]
        };
        (cond, uncond)
    }

    /// Encode a batch of text prompts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.encode_text(t)).collect()
    }
}

// ---------------------------------------------------------------------------
// TextBatch
// ---------------------------------------------------------------------------

/// A batch of text conditions for a single generation request.
pub struct TextBatch {
    pub texts: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
    pub seq_len: usize,
    pub embed_dim: usize,
}

impl TextBatch {
    /// Build a [`TextBatch`] by encoding `texts` with the provided conditioner.
    pub fn from_conditioner(conditioner: &TextConditioner, texts: &[&str]) -> Self {
        let embeddings = conditioner.encode_batch(texts);
        let seq_len = conditioner.sequence_length();
        let embed_dim = conditioner.embedding.embed_dim;
        Self {
            texts: texts.iter().map(|s| s.to_string()).collect(),
            embeddings,
            seq_len,
            embed_dim,
        }
    }

    /// Number of texts (and embeddings) in the batch.
    pub fn num_texts(&self) -> usize {
        self.texts.len()
    }

    /// Element-wise mean embedding across all texts in the batch.
    ///
    /// Returns an empty `Vec` if the batch is empty.
    pub fn mean_embedding(&self) -> Vec<f32> {
        if self.embeddings.is_empty() {
            return Vec::new();
        }
        let len = self.embeddings[0].len();
        let n = self.embeddings.len() as f32;
        let mut mean = vec![0.0_f32; len];
        for emb in &self.embeddings {
            for (m, &v) in mean.iter_mut().zip(emb.iter()) {
                *m += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        mean
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::collections::HashMap as StdHashMap;

    // --- SimpleTokenizer ---

    #[test]
    fn test_tokenizer_new() {
        let tok = SimpleTokenizer::new(&["hello", "world"]);
        assert_eq!(tok.pad_token_id, 0);
        assert_eq!(tok.unk_token_id, 1);
        assert_eq!(tok.bos_token_id, 2);
        assert_eq!(tok.eos_token_id, 3);
        // 4 special + 2 words
        assert_eq!(tok.vocab_size(), 6);
        assert_eq!(tok.max_length, 77);
    }

    #[test]
    fn test_tokenizer_with_max_length() {
        let tok = SimpleTokenizer::with_max_length(&["a"], 12);
        assert_eq!(tok.max_length, 12);
    }

    #[test]
    fn test_tokenizer_default_face_vocab_size() {
        let tok = SimpleTokenizer::default_face_vocab();
        // 4 special tokens + 34 face words
        assert_eq!(tok.vocab_size(), 4 + 34);
    }

    #[test]
    fn test_tokenize_simple_text() {
        let tok = SimpleTokenizer::new(&["face", "portrait"]);
        let tokens = tok.tokenize("face portrait");
        // [BOS=2, face_id, portrait_id, EOS=3]
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], tok.bos_token_id);
        assert_eq!(tokens[3], tok.eos_token_id);
    }

    #[test]
    fn test_tokenize_adds_bos_eos() {
        let tok = SimpleTokenizer::new(&["test"]);
        let tokens = tok.tokenize("test");
        assert_eq!(tokens.first().copied(), Some(tok.bos_token_id));
        assert_eq!(tokens.last().copied(), Some(tok.eos_token_id));
    }

    #[test]
    fn test_tokenize_unk_token() {
        let tok = SimpleTokenizer::new(&["known"]);
        let tokens = tok.tokenize("known unknown");
        // [BOS, known_id, UNK, EOS]
        assert_eq!(tokens[2], tok.unk_token_id);
    }

    /// Regression: truncation used to happen *after* EOS was appended, so long
    /// prompts silently lost their end-of-sequence marker.
    #[test]
    fn test_tokenize_keeps_eos_when_truncated() {
        let tok = SimpleTokenizer::with_max_length(&["face", "portrait", "smile"], 5);
        let tokens = tok.tokenize("face portrait smile face portrait smile face");
        assert_eq!(tokens.len(), 5, "must never exceed max_length");
        assert_eq!(tokens.first().copied(), Some(tok.bos_token_id));
        assert_eq!(
            tokens.last().copied(),
            Some(tok.eos_token_id),
            "a truncated prompt must still end with EOS"
        );
    }

    #[test]
    fn test_tokenize_degenerate_max_lengths() {
        let mut tok = SimpleTokenizer::new(&["face"]);
        tok.max_length = 0;
        assert!(tok.tokenize("face").is_empty());
        assert!(tok.encode_padded("face").is_empty());

        tok.max_length = 1;
        assert_eq!(tok.tokenize("face"), vec![tok.bos_token_id]);
        assert_eq!(tok.encode_padded("face").len(), 1);

        tok.max_length = 2;
        assert_eq!(
            tok.tokenize("face"),
            vec![tok.bos_token_id, tok.eos_token_id]
        );
    }

    #[test]
    fn test_encode_padded_length() {
        let tok = SimpleTokenizer::new(&["a", "b"]);
        let tokens = tok.encode_padded("a b");
        assert_eq!(tokens.len(), tok.max_length);
        // Tail should be PAD
        assert_eq!(tokens.last().copied(), Some(tok.pad_token_id));
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = SimpleTokenizer::new(&["face", "portrait"]);
        let tokens = tok.tokenize("face portrait");
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, "face portrait");
    }

    #[test]
    fn test_from_vocab_file_roundtrip() -> DiffusionResult<()> {
        let path = std::env::temp_dir().join("oxigaf_text_vocab_roundtrip.txt");
        std::fs::write(&path, "# comment\nFace\n\nportrait\n").map_err(|e| {
            DiffusionError::ModelLoad(format!("failed to write test vocabulary: {e}"))
        })?;

        let tok = SimpleTokenizer::from_vocab_file(&path, 8)?;
        let _ = std::fs::remove_file(&path);

        assert_eq!(tok.max_length, 8);
        // 4 special + face + portrait
        assert_eq!(tok.vocab_size(), 6);
        let tokens = tok.tokenize("face portrait");
        assert_eq!(tok.decode(&tokens), "face portrait");
        Ok(())
    }

    #[test]
    fn test_from_vocab_file_rejects_empty_file() -> DiffusionResult<()> {
        let path = std::env::temp_dir().join("oxigaf_text_vocab_empty.txt");
        std::fs::write(&path, "# only comments\n\n").map_err(|e| {
            DiffusionError::ModelLoad(format!("failed to write test vocabulary: {e}"))
        })?;
        let result = SimpleTokenizer::from_vocab_file(&path, 8);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(DiffusionError::InvalidConfig(_))));
        Ok(())
    }

    // --- TextEmbedding ---

    #[test]
    fn test_text_embedding_new() {
        let emb = TextEmbedding::new(10, 16, 123);
        assert_eq!(emb.vocab_size, 10);
        assert_eq!(emb.embed_dim, 16);
        assert_eq!(emb.weights.len(), 10 * 16);
        assert!(emb.is_placeholder());
        // Values should be in [-0.02, 0.02]
        for &w in &emb.weights {
            assert!((-0.02..=0.02).contains(&w), "weight {w} out of range");
        }
    }

    #[test]
    fn test_text_embedding_zeros() {
        let emb = TextEmbedding::zeros(5, 8);
        assert!(emb.weights.iter().all(|&w| w == 0.0));
        assert!(emb.is_placeholder());
    }

    #[test]
    fn test_lookup_valid_token() {
        let emb = TextEmbedding::zeros(4, 8);
        let slice = emb.lookup(2);
        assert_eq!(slice.len(), 8);
    }

    #[test]
    fn test_lookup_oob_token() {
        let emb = TextEmbedding::zeros(4, 8);
        // OOB — should return zero slice, not panic
        let slice = emb.lookup(100);
        assert_eq!(slice.len(), 8);
        assert!(slice.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_embed_sequence_shape() {
        let emb = TextEmbedding::zeros(10, 16);
        let tokens = vec![0u32, 1, 2, 3, 4];
        let embedded = emb.embed_sequence(&tokens);
        assert_eq!(embedded.len(), 5 * 16);
    }

    #[test]
    fn test_from_weights_validates_length() {
        let ok = TextEmbedding::from_weights(2, 3, vec![0.0; 6]);
        assert!(ok.is_ok());
        assert!(!ok.map(|e| e.is_placeholder()).unwrap_or(true));

        assert!(matches!(
            TextEmbedding::from_weights(2, 3, vec![0.0; 5]),
            Err(DiffusionError::InvalidConfig(_))
        ));
        assert!(matches!(
            TextEmbedding::from_weights(0, 3, vec![]),
            Err(DiffusionError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_from_safetensors_roundtrip() -> DiffusionResult<()> {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let tensor = Tensor::from_vec(values.clone(), (4, 3), &device)
            .map_err(|e| DiffusionError::ModelLoad(format!("test tensor: {e}")))?;

        let mut map: StdHashMap<String, Tensor> = StdHashMap::new();
        map.insert("token_embedding.weight".to_string(), tensor);

        let path = std::env::temp_dir().join("oxigaf_text_embedding_roundtrip.safetensors");
        candle_core::safetensors::save(&map, &path)
            .map_err(|e| DiffusionError::ModelLoad(format!("test save: {e}")))?;

        let loaded = TextEmbedding::from_safetensors(&path, "token_embedding.weight");
        let missing = TextEmbedding::from_safetensors(&path, "does.not.exist");
        let _ = std::fs::remove_file(&path);

        let loaded = loaded?;
        assert_eq!(loaded.vocab_size, 4);
        assert_eq!(loaded.embed_dim, 3);
        assert!(
            !loaded.is_placeholder(),
            "loaded weights are not a placeholder"
        );
        assert_eq!(loaded.lookup(1), &values[3..6]);
        assert!(matches!(missing, Err(DiffusionError::ModelLoad(_))));
        Ok(())
    }

    // --- TextConditioningConfig ---

    #[test]
    fn test_config_default() {
        let cfg = TextConditioningConfig::default();
        assert_eq!(cfg.embed_dim, 768);
        assert_eq!(cfg.max_length, 77);
        assert!((cfg.guidance_scale - 7.5).abs() < 1e-6);
        assert!(!cfg.use_negative_prompt);
    }

    #[test]
    fn test_config_validate() {
        let good = TextConditioningConfig::default();
        assert!(good.validate().is_ok());

        let bad = TextConditioningConfig {
            embed_dim: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad2 = TextConditioningConfig {
            guidance_scale: -1.0,
            ..Default::default()
        };
        assert!(bad2.validate().is_err());

        let bad3 = TextConditioningConfig {
            max_length: 0,
            ..Default::default()
        };
        assert!(bad3.validate().is_err());
    }

    // --- TextConditioner ---

    #[test]
    fn test_text_conditioner_encode_text() {
        let cfg = TextConditioningConfig::default();
        let conditioner = TextConditioner::new(cfg.clone());
        let emb = conditioner.encode_text("frontal portrait");
        assert_eq!(emb.len(), cfg.max_length * cfg.embed_dim);
    }

    /// Regression: `TextConditioner::new` used to leave the tokenizer at its
    /// hard-coded 77-token length, so `encode_text` and the zero-filled
    /// unconditional buffer had different lengths for any other `max_length`.
    #[test]
    fn test_conditioner_propagates_max_length() {
        let cfg = TextConditioningConfig {
            max_length: 32,
            embed_dim: 8,
            ..Default::default()
        };
        let conditioner = TextConditioner::new(cfg.clone());
        assert_eq!(conditioner.tokenizer.max_length, 32);
        assert_eq!(conditioner.sequence_length(), 32);
        assert_eq!(conditioner.embedding_len(), 32 * 8);
        assert_eq!(conditioner.encode_text("happy face").len(), 32 * 8);
    }

    #[test]
    fn test_encode_with_cfg_lengths_match_for_custom_max_length() {
        let cfg = TextConditioningConfig {
            max_length: 32,
            embed_dim: 8,
            ..Default::default()
        };
        let conditioner = TextConditioner::new(cfg.clone());
        let (cond, uncond) = conditioner.encode_with_cfg("happy face");
        assert_eq!(cond.len(), uncond.len());
        assert_eq!(cond.len(), 32 * 8);
    }

    #[test]
    fn test_text_conditioner_encode_with_cfg() {
        let cfg = TextConditioningConfig::default();
        let conditioner = TextConditioner::new(cfg.clone());
        let (cond, uncond) = conditioner.encode_with_cfg("happy face");
        assert_eq!(cond.len(), cfg.max_length * cfg.embed_dim);
        assert_eq!(uncond.len(), cfg.max_length * cfg.embed_dim);
        // Unconditional should be zeros when use_negative_prompt=false
        assert!(uncond.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_text_conditioner_encode_batch() {
        let cfg = TextConditioningConfig::default();
        let conditioner = TextConditioner::new(cfg.clone());
        let texts = &["frontal portrait", "sad expression", "profile view"];
        let batch = conditioner.encode_batch(texts);
        assert_eq!(batch.len(), 3);
        for emb in &batch {
            assert_eq!(emb.len(), cfg.max_length * cfg.embed_dim);
        }
    }

    #[test]
    fn test_try_new_rejects_invalid_config() {
        let cfg = TextConditioningConfig {
            max_length: 0,
            ..Default::default()
        };
        assert!(TextConditioner::try_new(cfg).is_err());
    }

    /// The placeholder table must announce itself so callers never mistake
    /// random vectors for a working text encoder.
    #[test]
    fn test_placeholder_flag_is_set_for_random_weights() {
        let conditioner = TextConditioner::new(TextConditioningConfig {
            embed_dim: 4,
            max_length: 8,
            ..Default::default()
        });
        assert!(conditioner.is_placeholder());
    }

    #[test]
    fn test_from_pretrained_loads_real_weights() -> DiffusionResult<()> {
        let device = Device::Cpu;
        let cfg = TextConditioningConfig {
            embed_dim: 3,
            max_length: 8,
            ..Default::default()
        };

        // 4 special tokens + 2 words = 6 rows.
        let values: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(values, (6, 3), &device)
            .map_err(|e| DiffusionError::ModelLoad(format!("test tensor: {e}")))?;
        let mut map: StdHashMap<String, Tensor> = StdHashMap::new();
        map.insert("weight".to_string(), tensor);

        let weights_path = std::env::temp_dir().join("oxigaf_text_pretrained.safetensors");
        let vocab_path = std::env::temp_dir().join("oxigaf_text_pretrained_vocab.txt");
        candle_core::safetensors::save(&map, &weights_path)
            .map_err(|e| DiffusionError::ModelLoad(format!("test save: {e}")))?;
        std::fs::write(&vocab_path, "face\nportrait\n")
            .map_err(|e| DiffusionError::ModelLoad(format!("test vocab write: {e}")))?;

        let conditioner =
            TextConditioner::from_pretrained(cfg.clone(), &weights_path, "weight", &vocab_path);

        // A mismatching embed_dim must be rejected.
        let mismatched = TextConditioner::from_pretrained(
            TextConditioningConfig {
                embed_dim: 16,
                ..cfg.clone()
            },
            &weights_path,
            "weight",
            &vocab_path,
        );

        let _ = std::fs::remove_file(&weights_path);
        let _ = std::fs::remove_file(&vocab_path);

        let conditioner = conditioner?;
        assert!(!conditioner.is_placeholder());
        assert_eq!(conditioner.sequence_length(), 8);
        assert_eq!(conditioner.encode_text("face portrait").len(), 8 * 3);
        assert!(matches!(mismatched, Err(DiffusionError::InvalidConfig(_))));
        Ok(())
    }

    // --- TextBatch ---

    #[test]
    fn test_text_batch_from_conditioner() {
        let cfg = TextConditioningConfig::default();
        let conditioner = TextConditioner::new(cfg.clone());
        let texts = &["face", "portrait"];
        let batch = TextBatch::from_conditioner(&conditioner, texts);
        assert_eq!(batch.num_texts(), 2);
        assert_eq!(batch.seq_len, cfg.max_length);
        assert_eq!(batch.embed_dim, cfg.embed_dim);
    }

    #[test]
    fn test_text_batch_mean_embedding() {
        let cfg = TextConditioningConfig::default();
        let conditioner = TextConditioner::new(cfg.clone());
        let texts = &["happy face", "sad face"];
        let batch = TextBatch::from_conditioner(&conditioner, texts);
        let mean = batch.mean_embedding();
        assert_eq!(mean.len(), cfg.max_length * cfg.embed_dim);

        // The mean of the two embeddings should lie between their extremes
        for (i, (&v0, &v1)) in batch.embeddings[0]
            .iter()
            .zip(batch.embeddings[1].iter())
            .enumerate()
        {
            let expected = (v0 + v1) / 2.0;
            let got = mean[i];
            assert!(
                (got - expected).abs() < 1e-5,
                "mean mismatch at {i}: {got} vs {expected}"
            );
        }
    }
}
