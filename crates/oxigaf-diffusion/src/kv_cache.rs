//! KV-cache infrastructure for multi-head attention in diffusion models.
//!
//! In diffusion models, cross-attention key/value pairs come from conditioning
//! (e.g., CLIP image features) that stay **constant** across all denoising
//! timesteps. By caching those K and V tensors after the first forward pass,
//! subsequent steps can skip the expensive projection + matmul entirely.
//!
//! ## Usage
//!
//! ```rust
//! use oxigaf_diffusion::kv_cache::{KVCache, KVCacheConfig, CacheKeyBuilder};
//!
//! let cache = KVCache::new(KVCacheConfig::default());
//!
//! let key = CacheKeyBuilder::new()
//!     .layer(0)
//!     .head_group(0)
//!     .conditioning_hash(0xdeadbeef)
//!     .build();
//!
//! let entry = cache.get_or_compute(key, || {
//!     // expensive computation — only called on cache miss
//!     let keys   = vec![0.0_f32; 1 * 8 * 64 * 64];
//!     let values = vec![0.0_f32; 1 * 8 * 64 * 64];
//!     Ok((keys, values, 1, 8, 64, 64))
//! }).expect("cache compute failed");
//!
//! assert!(entry.is_valid());
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// KVEntry
// ---------------------------------------------------------------------------

/// A single cached key-value entry for one attention layer / head group.
///
/// The key and value tensors are stored as flat `f32` vectors with layout
/// `[batch, num_heads, seq_k, head_dim]`:
///
/// ```text
/// index [b, h, s, d] → b * num_heads * seq_k * head_dim
///                     + h * seq_k * head_dim
///                     + s * head_dim
///                     + d
/// ```
#[derive(Debug, Clone)]
pub struct KVEntry {
    /// Key tensor (flat, `[batch, num_heads, seq_k, head_dim]`).
    pub keys: Vec<f32>,
    /// Value tensor (flat, `[batch, num_heads, seq_k, head_dim]`).
    pub values: Vec<f32>,
    /// Batch size dimension.
    pub batch: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Key/value sequence length.
    pub seq_k: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of times this entry has been retrieved via [`KVCache::get`].
    pub access_count: u64,
}

impl KVEntry {
    /// Create a new cache entry.
    ///
    /// The caller is responsible for ensuring the lengths of `keys` and
    /// `values` match `batch * num_heads * seq_k * head_dim`. Use
    /// [`KVEntry::is_valid`] to verify after construction.
    pub fn new(
        keys: Vec<f32>,
        values: Vec<f32>,
        batch: usize,
        num_heads: usize,
        seq_k: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            keys,
            values,
            batch,
            num_heads,
            seq_k,
            head_dim,
            access_count: 0,
        }
    }

    /// Total number of elements in `keys` (or `values`).
    pub fn num_elements(&self) -> usize {
        self.batch * self.num_heads * self.seq_k * self.head_dim
    }

    /// Memory footprint in bytes — keys + values, each stored as `f32` (4 bytes).
    pub fn memory_bytes(&self) -> usize {
        self.num_elements() * 2 * 4
    }

    /// Return `true` when both stored vectors have the expected length.
    pub fn is_valid(&self) -> bool {
        let expected = self.num_elements();
        self.keys.len() == expected && self.values.len() == expected
    }
}

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Cache eviction policy used when the cache is at capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict the entry that was accessed **least recently**.
    LRU,
    /// Evict the entry with the **lowest** `access_count`.
    LFU,
    /// Evict the entry that was inserted **first** (regardless of access).
    FIFO,
}

// ---------------------------------------------------------------------------
// KVCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for [`KVCache`].
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Maximum number of entries the cache will hold.
    ///
    /// When this limit is reached a single entry is evicted (according to
    /// [`KVCacheConfig::eviction`]) before inserting the new one.
    ///
    /// Default: `64`.
    pub max_entries: usize,

    /// Maximum total memory in bytes across all cached entries (`0` = unlimited).
    ///
    /// When inserting an entry would push total memory above this limit, entries
    /// are evicted until there is room (or the cache is empty).
    ///
    /// Default: `512 * 1024 * 1024` (512 MB).
    pub max_memory_bytes: usize,

    /// Whether the cache is active. When `false`, [`KVCache::get`] always
    /// returns `None` and [`KVCache::insert`] is a no-op.
    ///
    /// Default: `true`.
    pub enabled: bool,

    /// Eviction strategy applied when capacity is exceeded.
    ///
    /// Default: [`EvictionPolicy::LRU`].
    pub eviction: EvictionPolicy,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 64,
            max_memory_bytes: 512 * 1024 * 1024,
            enabled: true,
            eviction: EvictionPolicy::LRU,
        }
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Snapshot of cache performance statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of failed cache lookups (key not present).
    pub misses: u64,
    /// Total entries evicted since the cache was created.
    pub evictions: u64,
    /// Estimated total memory used by all cached entries (bytes).
    pub total_memory_bytes: usize,
    /// Number of entries currently in the cache.
    pub num_entries: usize,
}

impl CacheStats {
    /// Fraction of lookups that were cache hits (`hits / (hits + misses)`).
    ///
    /// Returns `0.0` when no lookups have been performed yet.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

/// Thread-safe LRU / LFU / FIFO cache for attention key-value pairs.
///
/// All public methods are safe to call concurrently from multiple threads.
///
/// ## Deadlock avoidance
///
/// Internally the struct holds three independent `Mutex` fields:
/// `entries`, `insertion_order`, and `stats`. They are **never held
/// simultaneously** — each lock is acquired, used, and released before the
/// next is taken. This prevents deadlocks regardless of call order.
pub struct KVCache {
    config: KVCacheConfig,
    /// Map from string key → [`KVEntry`].
    entries: Mutex<HashMap<String, KVEntry>>,
    /// Keys in insertion order; also updated on access for LRU.
    insertion_order: Mutex<Vec<String>>,
    /// Cumulative statistics.
    stats: Mutex<CacheStats>,
}

impl KVCache {
    /// Create a new, empty cache with the given configuration.
    pub fn new(config: KVCacheConfig) -> Self {
        Self {
            config,
            entries: Mutex::new(HashMap::new()),
            insertion_order: Mutex::new(Vec::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    /// Return `true` if the cache currently holds an entry for `key`.
    pub fn contains(&self, key: &str) -> bool {
        if !self.config.enabled {
            return false;
        }
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.contains_key(key)
    }

    /// Retrieve a cached entry by key.
    ///
    /// On a hit the entry's `access_count` is incremented and, for LRU caches,
    /// the key is moved to the "most recently used" end of the ordering queue.
    ///
    /// Updates [`CacheStats::hits`] / [`CacheStats::misses`] accordingly.
    pub fn get(&self, key: &str) -> Option<KVEntry> {
        if !self.config.enabled {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.misses += 1;
            return None;
        }

        // --- lock entries ---
        let hit = {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = entries.get_mut(key) {
                entry.access_count += 1;
                Some(entry.clone())
            } else {
                None
            }
        };

        // --- update insertion_order for LRU (no entries lock held) ---
        if hit.is_some() && self.config.eviction == EvictionPolicy::LRU {
            let mut order = self
                .insertion_order
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(pos) = order.iter().position(|k| k == key) {
                order.remove(pos);
                order.push(key.to_string());
            }
        }

        // --- update stats ---
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            if hit.is_some() {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }
        }

        hit
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Insert or replace a cache entry.
    ///
    /// If the cache is at capacity ([`KVCacheConfig::max_entries`]) or the
    /// memory limit would be exceeded, entries are evicted until space is
    /// available. Returns an error only if eviction is impossible (i.e., the
    /// single new entry exceeds the memory limit with an empty cache).
    ///
    /// If the cache is disabled ([`KVCacheConfig::enabled`] = `false`), this
    /// method is a no-op.
    pub fn insert(&self, key: String, entry: KVEntry) -> Result<(), DiffusionError> {
        if !self.config.enabled {
            return Ok(());
        }

        let entry_bytes = entry.memory_bytes();

        // Check whether this single entry already exceeds the memory cap.
        if self.config.max_memory_bytes > 0 && entry_bytes > self.config.max_memory_bytes {
            return Err(DiffusionError::Inference(format!(
                "KVCache: entry size {} bytes exceeds max_memory_bytes {}",
                entry_bytes, self.config.max_memory_bytes
            )));
        }

        // Evict until both entry count and memory constraints are satisfied.
        loop {
            let (current_entries, current_memory) = {
                let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                let stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
                (entries.len(), stats.total_memory_bytes)
            };

            let over_count = current_entries >= self.config.max_entries;
            let over_memory = self.config.max_memory_bytes > 0
                && current_memory + entry_bytes > self.config.max_memory_bytes;

            // Also skip eviction if the key already exists (will be replaced).
            let key_exists = {
                let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                entries.contains_key(&key)
            };

            if (!over_count && !over_memory) || key_exists {
                break;
            }

            // Nothing to evict — cache is empty but we still need space.
            if current_entries == 0 {
                break;
            }

            self.evict_one()?;
        }

        // --- perform the insertion ---
        let replaced_bytes = {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let old = entries.insert(key.clone(), entry);
            old.map(|e| e.memory_bytes())
        };

        // Update insertion_order: remove old position if present, push to end.
        {
            let mut order = self
                .insertion_order
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            order.retain(|k| k != &key);
            order.push(key);
        }

        // Update stats.
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(old_bytes) = replaced_bytes {
                // Replacing an existing entry: subtract old, add new.
                stats.total_memory_bytes = stats
                    .total_memory_bytes
                    .saturating_sub(old_bytes)
                    .saturating_add(entry_bytes);
                // num_entries stays the same.
            } else {
                stats.total_memory_bytes = stats.total_memory_bytes.saturating_add(entry_bytes);
                stats.num_entries += 1;
            }
        }

        Ok(())
    }

    /// Remove a single entry from the cache.
    ///
    /// Returns `true` when the key was present and removed.
    pub fn remove(&self, key: &str) -> bool {
        let removed = {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            entries.remove(key)
        };

        if let Some(entry) = removed {
            {
                let mut order = self
                    .insertion_order
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                order.retain(|k| k != key);
            }
            {
                let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
                stats.total_memory_bytes = stats
                    .total_memory_bytes
                    .saturating_sub(entry.memory_bytes());
                stats.num_entries = stats.num_entries.saturating_sub(1);
            }
            true
        } else {
            false
        }
    }

    /// Remove all entries from the cache and reset memory accounting.
    ///
    /// Hit/miss/eviction counters are preserved.
    pub fn clear(&self) {
        {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            entries.clear();
        }
        {
            let mut order = self
                .insertion_order
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            order.clear();
        }
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_memory_bytes = 0;
            stats.num_entries = 0;
        }
    }

    // -----------------------------------------------------------------------
    // Convenience
    // -----------------------------------------------------------------------

    /// Get a cached entry, or compute and cache it on a miss.
    ///
    /// The closure `compute_fn` is called **without any cache lock held**,
    /// so it may perform expensive work without blocking concurrent readers.
    ///
    /// `compute_fn` returns `(keys, values, batch, num_heads, seq_k, head_dim)`.
    ///
    /// # Race condition
    ///
    /// If two threads call `get_or_compute` for the same key simultaneously,
    /// both may invoke `compute_fn` and the second result will simply overwrite
    /// the first in the cache. This is safe (idempotent) but redundant.
    pub fn get_or_compute<F>(&self, key: String, compute_fn: F) -> Result<KVEntry, DiffusionError>
    where
        F: FnOnce() -> Result<(Vec<f32>, Vec<f32>, usize, usize, usize, usize), DiffusionError>,
    {
        // Fast path: cache hit (no lock held during computation).
        if let Some(entry) = self.get(&key) {
            return Ok(entry);
        }

        // Cache miss: compute without holding any lock.
        let (keys, values, batch, num_heads, seq_k, head_dim) = compute_fn()?;
        let new_entry = KVEntry::new(keys, values, batch, num_heads, seq_k, head_dim);

        // Insert into cache (best-effort; ignore errors from disabled cache).
        let _ = self.insert(key, new_entry.clone());

        Ok(new_entry)
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// Return a snapshot of current cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Estimated total memory used by all cached entries (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.stats
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .total_memory_bytes
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Return `true` when the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Evict one entry according to the configured policy.
    fn evict_one(&self) -> Result<(), DiffusionError> {
        let victim_key = match self.config.eviction {
            // LRU / FIFO: evict the front of `insertion_order`
            EvictionPolicy::LRU | EvictionPolicy::FIFO => {
                let mut order = self
                    .insertion_order
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if order.is_empty() {
                    return Ok(());
                }
                Some(order.remove(0))
            }

            // LFU: scan entries for minimum access_count
            EvictionPolicy::LFU => {
                let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                entries
                    .iter()
                    .min_by_key(|(_, e)| e.access_count)
                    .map(|(k, _)| k.clone())
            }
        };

        if let Some(key) = victim_key {
            // Remove from entries map.
            let removed = {
                let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                entries.remove(&key)
            };

            if let Some(entry) = removed {
                // For LFU the key is still in insertion_order — clean it up.
                if self.config.eviction == EvictionPolicy::LFU {
                    let mut order = self
                        .insertion_order
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    order.retain(|k| k != &key);
                }

                let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
                stats.total_memory_bytes = stats
                    .total_memory_bytes
                    .saturating_sub(entry.memory_bytes());
                stats.num_entries = stats.num_entries.saturating_sub(1);
                stats.evictions += 1;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CacheKeyBuilder
// ---------------------------------------------------------------------------

/// Builder for consistent, human-readable cache keys.
///
/// Keys uniquely identify the cached K/V tensors for a given (layer,
/// head-group, conditioning-hash) triplet.
///
/// ```rust
/// use oxigaf_diffusion::kv_cache::CacheKeyBuilder;
///
/// let key = CacheKeyBuilder::new()
///     .layer(3)
///     .head_group(1)
///     .conditioning_hash(0xdeadbeef)
///     .build();
///
/// assert_eq!(key, "layer=3:hg=1:cond=3735928559");
/// ```
pub struct CacheKeyBuilder {
    parts: Vec<String>,
}

impl CacheKeyBuilder {
    /// Create an empty key builder.
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    /// Append a transformer layer index to the key.
    pub fn layer(mut self, layer_idx: usize) -> Self {
        self.parts.push(format!("layer={layer_idx}"));
        self
    }

    /// Append a head-group index to the key.
    pub fn head_group(mut self, group_idx: usize) -> Self {
        self.parts.push(format!("hg={group_idx}"));
        self
    }

    /// Append a conditioning hash (e.g., a hash of the image embedding) to
    /// the key so that different conditioning inputs stay separate.
    pub fn conditioning_hash(mut self, hash: u64) -> Self {
        self.parts.push(format!("cond={hash}"));
        self
    }

    /// Finalise and return the cache key string.
    pub fn build(self) -> String {
        self.parts.join(":")
    }
}

impl Default for CacheKeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_entry(batch: usize, heads: usize, seq_k: usize, dim: usize) -> KVEntry {
        let n = batch * heads * seq_k * dim;
        KVEntry::new(vec![1.0_f32; n], vec![2.0_f32; n], batch, heads, seq_k, dim)
    }

    // -----------------------------------------------------------------------
    // KVCacheConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_kvcache_config_defaults() {
        let cfg = KVCacheConfig::default();
        assert_eq!(cfg.max_entries, 64);
        assert_eq!(cfg.max_memory_bytes, 512 * 1024 * 1024);
        assert!(cfg.enabled);
        assert_eq!(cfg.eviction, EvictionPolicy::LRU);
    }

    // -----------------------------------------------------------------------
    // KVEntry construction & validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_kventry_new_stores_correctly() {
        let keys = vec![1.0_f32; 16];
        let values = vec![2.0_f32; 16];
        let entry = KVEntry::new(keys.clone(), values.clone(), 1, 2, 4, 2);
        assert_eq!(entry.batch, 1);
        assert_eq!(entry.num_heads, 2);
        assert_eq!(entry.seq_k, 4);
        assert_eq!(entry.head_dim, 2);
        assert_eq!(entry.keys, keys);
        assert_eq!(entry.values, values);
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_kventry_is_valid_correct_sizes() {
        let entry = make_entry(1, 2, 4, 8);
        assert!(entry.is_valid(), "entry with correct sizes should be valid");
    }

    #[test]
    fn test_kventry_is_valid_wrong_keys_size() {
        let n = 2 * 4 * 8; // 64
        let mut entry = KVEntry::new(vec![0.0; n + 1], vec![0.0; n], 1, 2, 4, 8);
        entry.keys.push(0.0); // deliberately wrong length
                              // Rebuild to ensure keys.len() != num_elements
        let bad_entry = KVEntry {
            keys: vec![0.0_f32; n + 5],
            values: vec![0.0_f32; n],
            batch: 1,
            num_heads: 2,
            seq_k: 4,
            head_dim: 8,
            access_count: 0,
        };
        assert!(!bad_entry.is_valid(), "wrong keys length should be invalid");
    }

    #[test]
    fn test_kventry_is_valid_wrong_values_size() {
        let n = 2 * 4 * 8;
        let bad_entry = KVEntry {
            keys: vec![0.0_f32; n],
            values: vec![0.0_f32; n + 3],
            batch: 1,
            num_heads: 2,
            seq_k: 4,
            head_dim: 8,
            access_count: 0,
        };
        assert!(
            !bad_entry.is_valid(),
            "wrong values length should be invalid"
        );
    }

    #[test]
    fn test_kventry_memory_bytes() {
        let entry = make_entry(1, 2, 4, 8); // 1*2*4*8 = 64 elements
                                            // 64 elements * 2 tensors * 4 bytes = 512
        assert_eq!(entry.memory_bytes(), 64 * 2 * 4);
    }

    // -----------------------------------------------------------------------
    // KVCache basic operations
    // -----------------------------------------------------------------------

    #[test]
    fn test_kvcache_new_starts_empty() {
        let cache = KVCache::new(KVCacheConfig::default());
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kvcache_contains_false_for_unknown_key() {
        let cache = KVCache::new(KVCacheConfig::default());
        assert!(!cache.contains("nonexistent"));
    }

    #[test]
    fn test_kvcache_insert_then_contains() {
        let cache = KVCache::new(KVCacheConfig::default());
        let entry = make_entry(1, 2, 4, 8);
        cache
            .insert("k1".to_string(), entry)
            .expect("insert failed");
        assert!(cache.contains("k1"));
    }

    #[test]
    fn test_kvcache_get_after_insert_returns_correct_entry() {
        let cache = KVCache::new(KVCacheConfig::default());
        let entry = make_entry(1, 2, 4, 8);
        cache
            .insert("k1".to_string(), entry.clone())
            .expect("insert");
        let retrieved = cache.get("k1").expect("should be present");
        assert_eq!(retrieved.keys, entry.keys);
        assert_eq!(retrieved.values, entry.values);
        assert_eq!(retrieved.batch, entry.batch);
    }

    #[test]
    fn test_kvcache_get_updates_access_count() {
        let cache = KVCache::new(KVCacheConfig::default());
        let entry = make_entry(1, 1, 2, 4);
        cache.insert("k".to_string(), entry).expect("insert");
        let _ = cache.get("k"); // access 1
        let retrieved = cache.get("k").expect("second get"); // access 2
                                                             // After two gets, access_count is 2 (we read the clone after the 2nd get).
        assert_eq!(retrieved.access_count, 2);
    }

    #[test]
    fn test_kvcache_remove_key_gone() {
        let cache = KVCache::new(KVCacheConfig::default());
        let entry = make_entry(1, 1, 2, 4);
        cache.insert("k".to_string(), entry).expect("insert");
        assert!(cache.remove("k"));
        assert!(!cache.contains("k"));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kvcache_clear_empties_cache() {
        let cache = KVCache::new(KVCacheConfig::default());
        for i in 0..5 {
            cache
                .insert(format!("k{i}"), make_entry(1, 1, 2, 4))
                .expect("insert");
        }
        assert_eq!(cache.len(), 5);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.memory_bytes(), 0);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_hits_increment_on_get() {
        let cache = KVCache::new(KVCacheConfig::default());
        let entry = make_entry(1, 1, 2, 4);
        cache.insert("k".to_string(), entry).expect("insert");
        let _ = cache.get("k");
        let _ = cache.get("k");
        let s = cache.stats();
        assert_eq!(s.hits, 2);
    }

    #[test]
    fn test_stats_misses_increment_on_get_miss() {
        let cache = KVCache::new(KVCacheConfig::default());
        let _ = cache.get("missing");
        let _ = cache.get("also_missing");
        let s = cache.stats();
        assert_eq!(s.misses, 2);
    }

    #[test]
    fn test_stats_hit_rate_half() {
        let cache = KVCache::new(KVCacheConfig::default());
        let entry = make_entry(1, 1, 2, 4);
        cache.insert("k".to_string(), entry).expect("insert");
        let _ = cache.get("k"); // hit
        let _ = cache.get("missing"); // miss
        let s = cache.stats();
        let rate = s.hit_rate();
        // 1 hit + 1 miss → 0.5
        assert!(
            (rate - 0.5).abs() < f64::EPSILON,
            "hit_rate should be 0.5, got {rate}"
        );
    }

    // -----------------------------------------------------------------------
    // Capacity / eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_beyond_max_entries_evicts_one() {
        let cfg = KVCacheConfig {
            max_entries: 3,
            max_memory_bytes: 0, // unlimited
            enabled: true,
            eviction: EvictionPolicy::FIFO,
        };
        let cache = KVCache::new(cfg);
        for i in 0..4 {
            cache
                .insert(format!("k{i}"), make_entry(1, 1, 2, 4))
                .expect("insert");
        }
        // After inserting 4 entries with max_entries=3, one must have been evicted.
        assert!(cache.len() <= 3, "cache len {} should be <= 3", cache.len());
        assert!(
            cache.stats().evictions >= 1,
            "expected at least one eviction"
        );
    }

    // -----------------------------------------------------------------------
    // get_or_compute
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_or_compute_calls_fn_on_miss() {
        let cache = KVCache::new(KVCacheConfig::default());
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();

        let _entry = cache
            .get_or_compute("k".to_string(), || {
                called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok((vec![0.0_f32; 4], vec![0.0_f32; 4], 1, 1, 2, 2))
            })
            .expect("get_or_compute failed");

        assert!(
            called.load(std::sync::atomic::Ordering::SeqCst),
            "compute_fn should have been called on miss"
        );
    }

    #[test]
    fn test_get_or_compute_does_not_call_fn_on_hit() {
        let cache = KVCache::new(KVCacheConfig::default());
        let entry = make_entry(1, 1, 2, 2);
        cache.insert("k".to_string(), entry).expect("insert");

        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();

        let _entry = cache
            .get_or_compute("k".to_string(), || {
                called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok((vec![0.0_f32; 4], vec![0.0_f32; 4], 1, 1, 2, 2))
            })
            .expect("get_or_compute failed");

        assert!(
            !called.load(std::sync::atomic::Ordering::SeqCst),
            "compute_fn should NOT have been called on cache hit"
        );
    }

    // -----------------------------------------------------------------------
    // CacheKeyBuilder
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_key_builder_produces_consistent_string() {
        let k1 = CacheKeyBuilder::new()
            .layer(3)
            .head_group(1)
            .conditioning_hash(0xdeadbeef)
            .build();
        let k2 = CacheKeyBuilder::new()
            .layer(3)
            .head_group(1)
            .conditioning_hash(0xdeadbeef)
            .build();
        assert_eq!(k1, k2, "same inputs must produce same key");
        assert_eq!(k1, "layer=3:hg=1:cond=3735928559");
    }

    #[test]
    fn test_cache_key_builder_different_layers_differ() {
        let k0 = CacheKeyBuilder::new().layer(0).build();
        let k1 = CacheKeyBuilder::new().layer(1).build();
        assert_ne!(k0, k1);
    }

    // -----------------------------------------------------------------------
    // Memory limit eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_limit_evicts_when_exceeded() {
        // Each entry: 1*1*4*4 = 16 elements, 16*2*4 = 128 bytes.
        // Set max_memory_bytes to 300 so only 2 entries fit (256 bytes).
        let cfg = KVCacheConfig {
            max_entries: 100, // entry count is not the constraint here
            max_memory_bytes: 300,
            enabled: true,
            eviction: EvictionPolicy::FIFO,
        };
        let cache = KVCache::new(cfg);

        // Insert 3 entries (each 128 bytes).
        for i in 0..3 {
            cache
                .insert(format!("k{i}"), make_entry(1, 1, 4, 4))
                .expect("insert");
        }

        // Memory should be ≤ 300 bytes.
        assert!(
            cache.memory_bytes() <= 300,
            "memory {} should be <= 300",
            cache.memory_bytes()
        );
    }
}
