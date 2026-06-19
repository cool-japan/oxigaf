//! Text prompt conditioning for diffusion models.
//!
//! Provides a simple whitespace/punctuation tokenizer with a fixed vocabulary
//! and a learned embedding table (random-initialized) so that text prompts
//! (e.g. "frontal portrait, neutral expression") can drive generation without
//! requiring actual CLIP weights.

use std::collections::HashMap;

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
    pub max_length: usize,
}

impl SimpleTokenizer {
    /// Build a tokenizer from a slice of vocabulary words.
    ///
    /// IDs are assigned: 0=PAD, 1=UNK, 2=BOS, 3=EOS, then vocab words starting
    /// at 4.  Words are lower-cased before insertion.
    pub fn new(vocab_words: &[&str]) -> Self {
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
            max_length: 77,
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
    /// Returns `[BOS, w1, w2, …, EOS]` truncated to `max_length`.
    /// Unknown words are mapped to `unk_token_id`.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let words = Self::split_into_words(text);
        let mut tokens: Vec<u32> = Vec::with_capacity(words.len() + 2);

        tokens.push(self.bos_token_id);
        for word in &words {
            let id = self.vocab.get(word).copied().unwrap_or(self.unk_token_id);
            tokens.push(id);
        }
        tokens.push(self.eos_token_id);
        tokens.truncate(self.max_length);
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

/// Learned (random-initialized) text embedding look-up table.
///
/// Weights are stored as a flat `[vocab_size × embed_dim]` `f32` buffer;
/// `lookup(id)` returns a `&[f32]` slice of length `embed_dim`.
pub struct TextEmbedding {
    pub vocab_size: usize,
    pub embed_dim: usize,
    /// Flat weight buffer: `weights[id * embed_dim..(id+1) * embed_dim]`.
    weights: Vec<f32>,
    /// Zero row returned for out-of-bounds token IDs.
    zero_row: Vec<f32>,
}

impl TextEmbedding {
    /// Create a new embedding table with xorshift64 random initialisation.
    ///
    /// Values are drawn uniformly from `[-0.02, 0.02]`.
    /// If `seed` is 0 the guard value `0xDEAD_BEEF` is used instead.
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
        }
    }

    /// Create an all-zeros embedding table.
    pub fn zeros(vocab_size: usize, embed_dim: usize) -> Self {
        let total = vocab_size * embed_dim;
        let zero_row = vec![0.0_f32; embed_dim];
        Self {
            vocab_size,
            embed_dim,
            weights: vec![0.0_f32; total],
            zero_row,
        }
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
        // Safety: bounds checked above.
        &self.weights[start..end]
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

/// End-to-end text conditioning pipeline for a diffusion model.
///
/// Combines a [`SimpleTokenizer`] and a [`TextEmbedding`] table to convert
/// raw text strings into dense embeddings suitable for cross-attention.
pub struct TextConditioner {
    pub tokenizer: SimpleTokenizer,
    pub embedding: TextEmbedding,
    pub config: TextConditioningConfig,
}

impl TextConditioner {
    /// Construct a conditioner from `config`.
    ///
    /// Uses the default face vocabulary and a randomly-initialized embedding
    /// table whose seed is derived from a fixed constant.
    pub fn new(config: TextConditioningConfig) -> Self {
        let tokenizer = SimpleTokenizer::default_face_vocab();
        let vocab_size = tokenizer.vocab_size();
        let embedding = TextEmbedding::new(vocab_size, config.embed_dim, 42);
        Self {
            tokenizer,
            embedding,
            config,
        }
    }

    /// Encode a single text prompt.
    ///
    /// Returns a flat buffer of shape `[max_length × embed_dim]`.
    pub fn encode_text(&self, text: &str) -> Vec<f32> {
        let tokens = self.tokenizer.encode_padded(text);
        self.embedding.embed_sequence(&tokens)
    }

    /// Encode text for classifier-free guidance.
    ///
    /// Returns `(conditional_embedding, unconditional_embedding)`.
    /// The unconditional embedding is either the encoding of
    /// `config.negative_prompt` (when `use_negative_prompt` is true) or an
    /// all-zeros buffer of the same shape.
    pub fn encode_with_cfg(&self, text: &str) -> (Vec<f32>, Vec<f32>) {
        let cond = self.encode_text(text);
        let uncond = if self.config.use_negative_prompt {
            self.encode_text(&self.config.negative_prompt.clone())
        } else {
            vec![0.0_f32; self.config.max_length * self.config.embed_dim]
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
        let seq_len = conditioner.config.max_length;
        let embed_dim = conditioner.config.embed_dim;
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

    // --- TextEmbedding ---

    #[test]
    fn test_text_embedding_new() {
        let emb = TextEmbedding::new(10, 16, 123);
        assert_eq!(emb.vocab_size, 10);
        assert_eq!(emb.embed_dim, 16);
        assert_eq!(emb.weights.len(), 10 * 16);
        // Values should be in [-0.02, 0.02]
        for &w in &emb.weights {
            assert!((-0.02..=0.02).contains(&w), "weight {w} out of range");
        }
    }

    #[test]
    fn test_text_embedding_zeros() {
        let emb = TextEmbedding::zeros(5, 8);
        assert!(emb.weights.iter().all(|&w| w == 0.0));
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
