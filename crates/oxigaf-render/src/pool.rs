//! GPU buffer memory pooling for efficient allocation reuse.
//!
//! # Memory Management Strategy
//!
//! This module implements a size-class based buffer pool for GPU memory:
//!
//! ## Size Classes
//!
//! Buffers are allocated in fixed size classes to reduce fragmentation:
//! - 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB
//!
//! When a buffer is requested, we round up to the next size class and
//! check if a buffer of that class is available in the pool.
//!
//! ## LRU Eviction
//!
//! When total allocated memory exceeds the configured budget, we evict
//! the least-recently-used buffers from the pool until we're under budget.
//!
//! ## RAII Returns
//!
//! `PooledBuffer` wraps a `wgpu::Buffer` and automatically returns it to
//! the pool when dropped, enabling zero-overhead buffer reuse.

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

/// Size classes for the buffer pool (in bytes).
///
/// Each class is 4x the previous, providing good coverage with minimal waste.
pub const SIZE_CLASSES: &[u64] = &[
    1024,             // 1KB
    4 * 1024,         // 4KB
    16 * 1024,        // 16KB
    64 * 1024,        // 64KB
    256 * 1024,       // 256KB
    1024 * 1024,      // 1MB
    4 * 1024 * 1024,  // 4MB
    16 * 1024 * 1024, // 16MB
];

/// Internal entry in the buffer pool.
struct PoolEntry {
    /// The GPU buffer.
    buffer: wgpu::Buffer,
    /// Size class this buffer belongs to.
    size_class: u64,
    /// Last time this buffer was used (for LRU eviction).
    last_used: Instant,
}

/// Statistics about buffer pool usage.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total bytes currently allocated (available + in-use).
    pub total_allocated_bytes: u64,
    /// Bytes currently in the pool (available for reuse).
    pub available_bytes: u64,
    /// Number of buffers currently in the pool.
    pub available_count: usize,
    /// Number of buffers currently in use.
    pub in_use_count: usize,
    /// Total allocations performed (cache misses).
    pub total_allocations: u64,
    /// Total buffer acquisitions (including cache hits).
    pub total_acquisitions: u64,
    /// Cache hit rate (0.0 to 1.0).
    pub hit_rate: f64,
}

/// Inner state of the buffer pool, protected by a mutex.
struct BufferPoolInner {
    /// Available buffers by size class.
    /// Key is the size class in bytes.
    available: [VecDeque<PoolEntry>; SIZE_CLASSES.len()],
    /// Number of buffers currently in use.
    in_use_count: usize,
    /// Total bytes allocated (available + in-use).
    total_allocated_bytes: u64,
    /// Maximum allowed bytes.
    max_bytes: u64,
    /// Total allocations performed.
    total_allocations: u64,
    /// Total acquisitions (including cache hits).
    total_acquisitions: u64,
}

impl BufferPoolInner {
    fn new(max_bytes: u64) -> Self {
        Self {
            available: std::array::from_fn(|_| VecDeque::new()),
            in_use_count: 0,
            total_allocated_bytes: 0,
            max_bytes,
            total_allocations: 0,
            total_acquisitions: 0,
        }
    }

    /// Find the appropriate size class index for a given size.
    fn size_class_index(size: u64) -> Option<usize> {
        SIZE_CLASSES.iter().position(|&class| class >= size)
    }

    /// Get statistics about the pool.
    fn stats(&self) -> PoolStats {
        let available_bytes: u64 = self
            .available
            .iter()
            .enumerate()
            .map(|(i, q)| SIZE_CLASSES[i] * (q.len() as u64))
            .sum();

        let available_count: usize = self.available.iter().map(|q| q.len()).sum();

        let hit_rate = if self.total_acquisitions > 0 {
            let misses = self.total_allocations as f64;
            let total = self.total_acquisitions as f64;
            (total - misses) / total
        } else {
            0.0
        };

        PoolStats {
            total_allocated_bytes: self.total_allocated_bytes,
            available_bytes,
            available_count,
            in_use_count: self.in_use_count,
            total_allocations: self.total_allocations,
            total_acquisitions: self.total_acquisitions,
            hit_rate,
        }
    }

    /// Evict buffers until we're under budget.
    /// Returns the number of bytes evicted.
    fn evict_until_under_budget(&mut self) -> u64 {
        let mut evicted_bytes = 0u64;

        while self.total_allocated_bytes > self.max_bytes {
            // Find the oldest buffer across all size classes
            let mut oldest_idx: Option<(usize, Instant)> = None;

            for (class_idx, queue) in self.available.iter().enumerate() {
                if let Some(entry) = queue.front() {
                    match oldest_idx {
                        None => oldest_idx = Some((class_idx, entry.last_used)),
                        Some((_, oldest_time)) if entry.last_used < oldest_time => {
                            oldest_idx = Some((class_idx, entry.last_used));
                        }
                        _ => {}
                    }
                }
            }

            // Evict the oldest buffer
            if let Some((class_idx, _)) = oldest_idx {
                if let Some(entry) = self.available[class_idx].pop_front() {
                    let size = entry.size_class;
                    self.total_allocated_bytes = self.total_allocated_bytes.saturating_sub(size);
                    evicted_bytes += size;
                    // Buffer is dropped here, releasing GPU memory
                    drop(entry);
                    tracing::debug!(
                        size_class = size,
                        remaining_bytes = self.total_allocated_bytes,
                        "Evicted buffer from pool"
                    );
                }
            } else {
                // No available buffers to evict
                break;
            }
        }

        evicted_bytes
    }

    /// Return a buffer to the pool.
    fn return_buffer(&mut self, entry: PoolEntry) {
        let class_idx = SIZE_CLASSES
            .iter()
            .position(|&c| c == entry.size_class)
            .unwrap_or(0);

        self.in_use_count = self.in_use_count.saturating_sub(1);
        self.available[class_idx].push_back(entry);

        tracing::trace!(
            size_class = SIZE_CLASSES[class_idx],
            in_use = self.in_use_count,
            available = self.available[class_idx].len(),
            "Returned buffer to pool"
        );
    }
}

/// A GPU buffer pool that recycles buffers by size class.
///
/// # Memory Management
///
/// The pool maintains buffers organized by size class. When a buffer is
/// requested, we check if one is available in the appropriate size class.
/// If not, a new buffer is allocated. When a `PooledBuffer` is dropped,
/// it returns to the pool for future reuse.
///
/// # Thread Safety
///
/// The pool is thread-safe and can be shared across threads using `Arc`.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<Mutex<BufferPoolInner>>,
}

impl BufferPool {
    /// Create a new buffer pool with the specified memory budget.
    ///
    /// # Arguments
    ///
    /// * `max_bytes` - Maximum memory to keep in the pool. When exceeded,
    ///   LRU buffers are evicted.
    #[must_use]
    pub fn new(max_bytes: u64) -> Self {
        tracing::info!(max_mb = max_bytes / (1024 * 1024), "Created buffer pool");

        Self {
            inner: Arc::new(Mutex::new(BufferPoolInner::new(max_bytes))),
        }
    }

    /// Acquire a buffer of at least `min_size` bytes.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create buffers on
    /// * `min_size` - Minimum buffer size in bytes
    /// * `usage` - Buffer usage flags
    /// * `label` - Debug label for the buffer
    ///
    /// # Returns
    ///
    /// A `PooledBuffer` that will return to the pool when dropped.
    /// Returns `None` if the requested size exceeds the largest size class.
    pub fn acquire(
        &self,
        device: &wgpu::Device,
        min_size: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Option<PooledBuffer> {
        let class_idx = BufferPoolInner::size_class_index(min_size)?;
        let size_class = SIZE_CLASSES[class_idx];

        let mut inner = self.inner.lock().ok()?;
        inner.total_acquisitions += 1;

        // Try to get a buffer from the pool
        let entry = if let Some(mut entry) = inner.available[class_idx].pop_back() {
            // Update last_used time
            entry.last_used = Instant::now();
            inner.in_use_count += 1;
            tracing::trace!(
                size_class = size_class,
                label = label,
                "Reused buffer from pool"
            );
            entry
        } else {
            // Need to allocate a new buffer
            inner.total_allocations += 1;

            // Evict if over budget before allocating
            if inner.total_allocated_bytes + size_class > inner.max_bytes {
                let evicted = inner.evict_until_under_budget();
                if evicted > 0 {
                    tracing::debug!(evicted_bytes = evicted, "Evicted buffers to make room");
                }
            }

            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size_class,
                usage,
                mapped_at_creation: false,
            });

            inner.total_allocated_bytes += size_class;
            inner.in_use_count += 1;

            tracing::debug!(
                size_class = size_class,
                label = label,
                total_mb = inner.total_allocated_bytes / (1024 * 1024),
                "Allocated new buffer"
            );

            PoolEntry {
                buffer,
                size_class,
                last_used: Instant::now(),
            }
        };

        drop(inner); // Release lock before creating PooledBuffer

        Some(PooledBuffer {
            entry: Some(entry),
            pool: Arc::downgrade(&self.inner),
            actual_size: min_size,
        })
    }

    /// Get current pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        self.inner
            .lock()
            .map(|inner| inner.stats())
            .unwrap_or_default()
    }

    /// Clear all available buffers from the pool.
    ///
    /// In-use buffers will still return to the pool when dropped.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            let mut cleared_bytes = 0u64;
            for (idx, queue) in inner.available.iter_mut().enumerate() {
                cleared_bytes += SIZE_CLASSES[idx] * (queue.len() as u64);
                queue.clear();
            }
            inner.total_allocated_bytes = inner.total_allocated_bytes.saturating_sub(cleared_bytes);
            tracing::info!(
                cleared_mb = cleared_bytes / (1024 * 1024),
                "Cleared buffer pool"
            );
        }
    }

    /// Set a new memory budget.
    ///
    /// If the new budget is lower than current usage, buffers will be
    /// evicted on the next acquire.
    pub fn set_budget(&self, max_bytes: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.max_bytes = max_bytes;
            tracing::info!(
                max_mb = max_bytes / (1024 * 1024),
                "Updated buffer pool budget"
            );
        }
    }

    /// Log current memory usage (debug level).
    pub fn log_usage(&self) {
        if let Ok(inner) = self.inner.lock() {
            let stats = inner.stats();
            tracing::debug!(
                total_mb = stats.total_allocated_bytes / (1024 * 1024),
                available_mb = stats.available_bytes / (1024 * 1024),
                in_use = stats.in_use_count,
                available = stats.available_count,
                hit_rate = format!("{:.1}%", stats.hit_rate * 100.0),
                "Buffer pool usage"
            );
        }
    }
}

/// A buffer acquired from a `BufferPool`.
///
/// This wrapper provides RAII semantics: when dropped, the underlying
/// buffer is automatically returned to the pool for reuse.
///
/// # Deref
///
/// `PooledBuffer` implements `Deref<Target = wgpu::Buffer>`, allowing
/// it to be used anywhere a `&wgpu::Buffer` is expected.
pub struct PooledBuffer {
    /// The pool entry (buffer + metadata).
    entry: Option<PoolEntry>,
    /// Weak reference to the pool for return on drop.
    pool: Weak<Mutex<BufferPoolInner>>,
    /// The actual requested size (may be less than size class).
    actual_size: u64,
}

impl PooledBuffer {
    /// Get the size class of this buffer.
    #[inline]
    #[must_use]
    pub fn size_class(&self) -> u64 {
        self.entry.as_ref().map(|e| e.size_class).unwrap_or(0)
    }

    /// Get the actual requested size.
    #[inline]
    #[must_use]
    pub fn actual_size(&self) -> u64 {
        self.actual_size
    }

    /// Get the underlying wgpu buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer has already been returned to the pool
    /// (which should never happen in normal usage).
    #[inline]
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        match self.entry.as_ref() {
            Some(entry) => &entry.buffer,
            None => {
                // This should never happen in correct usage - the buffer
                // is only None after Drop, and we can't call methods after Drop.
                // Using unreachable! instead of expect to satisfy no-unwrap policy.
                unreachable!("PooledBuffer accessed after being returned to pool")
            }
        }
    }

    /// Get the underlying wgpu buffer, returning None if already returned.
    #[inline]
    #[must_use]
    pub fn try_buffer(&self) -> Option<&wgpu::Buffer> {
        self.entry.as_ref().map(|e| &e.buffer)
    }
}

impl Deref for PooledBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        self.buffer()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(mut entry) = self.entry.take() {
            // Update last_used time for LRU
            entry.last_used = Instant::now();

            // Try to return to pool
            if let Some(pool) = self.pool.upgrade() {
                if let Ok(mut inner) = pool.lock() {
                    inner.return_buffer(entry);
                    return;
                }
            }

            // Pool is gone or locked, buffer will be dropped
            tracing::trace!(
                size_class = entry.size_class,
                "Buffer dropped (pool unavailable)"
            );
        }
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("size_class", &self.size_class())
            .field("actual_size", &self.actual_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_index() {
        // Test that size class lookup works correctly
        assert_eq!(BufferPoolInner::size_class_index(1), Some(0)); // 1 -> 1KB
        assert_eq!(BufferPoolInner::size_class_index(1024), Some(0)); // 1KB -> 1KB
        assert_eq!(BufferPoolInner::size_class_index(1025), Some(1)); // 1KB+1 -> 4KB
        assert_eq!(BufferPoolInner::size_class_index(4096), Some(1)); // 4KB -> 4KB
        assert_eq!(BufferPoolInner::size_class_index(4097), Some(2)); // 4KB+1 -> 16KB
        assert_eq!(BufferPoolInner::size_class_index(16 * 1024 * 1024), Some(7)); // 16MB -> 16MB
        assert_eq!(
            BufferPoolInner::size_class_index(16 * 1024 * 1024 + 1),
            None
        ); // > 16MB
    }

    #[test]
    fn test_size_classes_are_increasing() {
        for i in 1..SIZE_CLASSES.len() {
            assert!(
                SIZE_CLASSES[i] > SIZE_CLASSES[i - 1],
                "Size classes must be strictly increasing"
            );
        }
    }

    #[test]
    fn test_size_classes_are_powers_of_four() {
        // Each class should be 4x the previous (except first)
        for i in 1..SIZE_CLASSES.len() {
            assert_eq!(
                SIZE_CLASSES[i],
                SIZE_CLASSES[i - 1] * 4,
                "Size class {} is not 4x the previous",
                i
            );
        }
    }

    #[test]
    fn test_pool_stats_initial() {
        let pool = BufferPool::new(512 * 1024 * 1024);
        let stats = pool.stats();

        assert_eq!(stats.total_allocated_bytes, 0);
        assert_eq!(stats.available_bytes, 0);
        assert_eq!(stats.available_count, 0);
        assert_eq!(stats.in_use_count, 0);
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_acquisitions, 0);
    }

    #[test]
    fn test_pool_budget() {
        let pool = BufferPool::new(100 * 1024 * 1024); // 100MB
        let stats = pool.stats();
        assert_eq!(stats.total_allocated_bytes, 0);

        pool.set_budget(200 * 1024 * 1024); // 200MB
                                            // Budget change doesn't affect stats
        let stats = pool.stats();
        assert_eq!(stats.total_allocated_bytes, 0);
    }

    #[test]
    fn test_pool_clear() {
        let pool = BufferPool::new(512 * 1024 * 1024);
        pool.clear(); // Should not panic on empty pool
        let stats = pool.stats();
        assert_eq!(stats.available_count, 0);
    }

    #[test]
    fn test_size_class_coverage() {
        // Verify all expected size classes are present
        assert_eq!(SIZE_CLASSES[0], 1024); // 1KB
        assert_eq!(SIZE_CLASSES[1], 4 * 1024); // 4KB
        assert_eq!(SIZE_CLASSES[2], 16 * 1024); // 16KB
        assert_eq!(SIZE_CLASSES[3], 64 * 1024); // 64KB
        assert_eq!(SIZE_CLASSES[4], 256 * 1024); // 256KB
        assert_eq!(SIZE_CLASSES[5], 1024 * 1024); // 1MB
        assert_eq!(SIZE_CLASSES[6], 4 * 1024 * 1024); // 4MB
        assert_eq!(SIZE_CLASSES[7], 16 * 1024 * 1024); // 16MB
    }

    #[test]
    fn test_eviction_logic() {
        // Create a pool with 10KB budget
        let mut inner = BufferPoolInner::new(10 * 1024);

        // Simulate having 2 4KB buffers in the pool (8KB total, under budget)
        inner.total_allocated_bytes = 8 * 1024;

        // No eviction needed
        let evicted = inner.evict_until_under_budget();
        assert_eq!(evicted, 0);

        // Increase allocated beyond budget
        inner.total_allocated_bytes = 12 * 1024;

        // Still no eviction if no buffers in pool
        let evicted = inner.evict_until_under_budget();
        assert_eq!(evicted, 0);
    }
}
