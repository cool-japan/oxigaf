//! GPU buffer memory pooling for efficient allocation reuse.
//!
//! # Memory Management Strategy
//!
//! This module implements a size-class based buffer pool for GPU memory:
//!
//! ## Size Classes
//!
//! Buffers are allocated in fixed size classes to reduce fragmentation:
//! - 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB
//!
//! When a buffer is requested, we round up to the next size class and
//! check if a buffer of that class, whose usage flags are a superset of
//! the request, is available in the pool. A request larger than the
//! largest size class is still served: an exact-size buffer is allocated
//! as a one-off and is not returned to a free list on drop (there is no
//! size class for it to rejoin), so it is never reused.
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
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Instant;

/// Lock `mutex` for [`PooledBuffer`]'s `Drop` fallback, recovering the inner
/// guard if the mutex was poisoned.
///
/// This is deliberately used only on the `Drop::drop` return-to-pool path,
/// not on `BufferPool`'s other lock sites: `return_buffer`'s own operations
/// are simple, self-contained, non-panicking field updates, so recovering
/// there cannot compound whatever inconsistency caused the poisoning.
/// Silently skipping this step instead (the previous behaviour) would leave
/// `in_use_count` and `total_allocated_bytes` permanently overcounted from
/// that point on, since every subsequent `lock()` on a poisoned mutex also
/// returns `Err`. Other call sites keep failing closed on poison.
fn lock_recover(mutex: &Mutex<BufferPoolInner>) -> MutexGuard<'_, BufferPoolInner> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Size classes for the buffer pool (in bytes).
///
/// Each class is 4x the previous, providing good coverage with minimal waste.
pub const SIZE_CLASSES: &[u64] = &[
    1024,              // 1KB
    4 * 1024,          // 4KB
    16 * 1024,         // 16KB
    64 * 1024,         // 64KB
    256 * 1024,        // 256KB
    1024 * 1024,       // 1MB
    4 * 1024 * 1024,   // 4MB
    16 * 1024 * 1024,  // 16MB
    64 * 1024 * 1024,  // 64MB
    256 * 1024 * 1024, // 256MB
];

/// Internal entry in the buffer pool.
struct PoolEntry {
    /// The GPU buffer.
    buffer: wgpu::Buffer,
    /// Size class this buffer belongs to (or, for a one-off allocation that
    /// exceeded the largest size class, the exact requested size -- see
    /// [`BufferPoolInner::return_buffer`]).
    size_class: u64,
    /// The `wgpu::BufferUsages` this buffer was created with. A cache hit
    /// must only reuse an entry whose usage is a superset of what the new
    /// request needs, otherwise the buffer fails wgpu validation at
    /// bind/copy time.
    usage: wgpu::BufferUsages,
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
}

/// How a request should be satisfied, given the device's `max_buffer_size`.
///
/// Rounding a request up to its size class is what makes the pool reusable,
/// but the rounded-up size is a real `wgpu::Buffer` size and is therefore
/// subject to `wgpu::Limits::max_buffer_size`. Creating a buffer past that
/// limit is a validation error, which wgpu reports through the device's
/// uncaptured-error handler -- i.e. it *aborts the process* rather than
/// returning an error the pool could propagate. So the decision has to be
/// made before `create_buffer` is ever called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationPlan {
    /// Allocate (or reuse) `SIZE_CLASSES[idx]` bytes.
    SizeClass(usize),
    /// Allocate exactly the requested size as a one-off: either the request
    /// is above the largest size class, or its size class would exceed the
    /// device limit even though the request itself fits. Such a buffer has
    /// no free list to rejoin and is released on drop.
    Exact,
    /// The request exceeds `max_buffer_size`: it cannot be served at all on
    /// this device.
    TooLarge,
}

impl BufferPoolInner {
    /// Decide how to serve a request of `min_size` bytes on a device whose
    /// buffer-size limit is `max_buffer_size`.
    ///
    /// Pure function of its inputs so the (device-specific) policy can be
    /// unit-tested without a GPU.
    fn plan_allocation(min_size: u64, max_buffer_size: u64) -> AllocationPlan {
        if min_size > max_buffer_size {
            return AllocationPlan::TooLarge;
        }
        match Self::size_class_index(min_size) {
            // Size class fits within the device limit: pool it normally.
            Some(idx) if SIZE_CLASSES[idx] <= max_buffer_size => AllocationPlan::SizeClass(idx),
            // Either no class covers the request, or the covering class is
            // itself over the limit while the request is not. Both are
            // served exactly.
            _ => AllocationPlan::Exact,
        }
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
    ///
    /// A buffer whose `size_class` matches one of [`SIZE_CLASSES`] rejoins
    /// that class's free list for later reuse. A buffer allocated as a
    /// one-off (its requested size exceeded the largest size class -- see
    /// [`BufferPool::acquire`]) has no free list to rejoin: its allocation
    /// accounting is released here and the buffer itself is simply dropped.
    fn return_buffer(&mut self, entry: PoolEntry) {
        self.in_use_count = self.in_use_count.saturating_sub(1);

        match SIZE_CLASSES.iter().position(|&c| c == entry.size_class) {
            Some(class_idx) => {
                let size_class = entry.size_class;
                self.available[class_idx].push_back(entry);

                tracing::trace!(
                    size_class = size_class,
                    in_use = self.in_use_count,
                    available = self.available[class_idx].len(),
                    "Returned buffer to pool"
                );
            }
            None => {
                // One-off allocation that exceeded the largest size class:
                // it was never inserted into a free list, so release its
                // allocation accounting and let the buffer drop here.
                self.total_allocated_bytes =
                    self.total_allocated_bytes.saturating_sub(entry.size_class);

                tracing::trace!(
                    size = entry.size_class,
                    in_use = self.in_use_count,
                    "Dropped oversized one-off buffer (not pooled)"
                );
            }
        }
    }

    /// Remove and return the available entry in `class_idx` whose usage
    /// flags are a superset of `usage`, if one exists.
    ///
    /// A buffer created with usage flags `U` may only be used for
    /// operations covered by `U`; handing back an entry whose flags do not
    /// contain the request would fail wgpu validation at bind/copy time.
    /// Searches from the most-recently-returned entry first, preserving the
    /// LRU-friendly ordering the old unconditional `pop_back` provided.
    fn take_matching(&mut self, class_idx: usize, usage: wgpu::BufferUsages) -> Option<PoolEntry> {
        let queue = &mut self.available[class_idx];
        let pos = queue
            .iter()
            .rposition(|entry| entry.usage.contains(usage))?;
        queue.remove(pos)
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
    /// A `PooledBuffer` that will return to the pool when dropped. A cache
    /// hit only reuses a pooled buffer whose usage flags are a superset of
    /// `usage` (a buffer created for a narrower set of operations would
    /// fail wgpu validation if handed back for a wider one); otherwise a
    /// fresh buffer is allocated with exactly the requested `usage`.
    ///
    /// A request larger than the largest size class is still served, via a
    /// one-off exact-size allocation that is not returned to a free list on
    /// drop (there is no size class for it to rejoin) -- so size alone no
    /// longer causes this to return `None`.
    ///
    /// `None` is returned when the pool's internal mutex is poisoned, or when
    /// `min_size` exceeds the device's `max_buffer_size` limit -- the latter
    /// cannot be served by any allocation strategy, and creating the buffer
    /// anyway would trip wgpu validation, which aborts the process through
    /// the uncaptured-error handler instead of returning a recoverable error.
    /// A request that fits the device but whose *size class* would not is
    /// downgraded to an exact-size one-off allocation rather than refused.
    pub fn acquire(
        &self,
        device: &wgpu::Device,
        min_size: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Option<PooledBuffer> {
        let max_buffer_size = device.limits().max_buffer_size;
        let plan = BufferPoolInner::plan_allocation(min_size, max_buffer_size);
        if plan == AllocationPlan::TooLarge {
            tracing::error!(
                requested = min_size,
                max_buffer_size,
                label = label,
                "Buffer request exceeds the device's maximum buffer size; cannot be served"
            );
            return None;
        }

        let mut inner = self.inner.lock().ok()?;
        inner.total_acquisitions += 1;

        let entry = match plan {
            AllocationPlan::SizeClass(class_idx) => {
                let size_class = SIZE_CLASSES[class_idx];

                if let Some(mut entry) = inner.take_matching(class_idx, usage) {
                    // Cache hit: reuse a pooled buffer whose usage flags
                    // cover this request.
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
                            tracing::debug!(
                                evicted_bytes = evicted,
                                "Evicted buffers to make room"
                            );
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
                        usage,
                        last_used: Instant::now(),
                    }
                }
            }
            AllocationPlan::Exact => {
                // Either larger than the largest size class, or its size
                // class would exceed the device's `max_buffer_size` while
                // the request itself fits. Serve it as a one-off exact-size
                // allocation instead of failing the request.
                // `return_buffer` recognizes that its size does not match
                // any class and releases it on drop rather than pooling it.
                inner.total_allocations += 1;

                if inner.total_allocated_bytes + min_size > inner.max_bytes {
                    let evicted = inner.evict_until_under_budget();
                    if evicted > 0 {
                        tracing::debug!(evicted_bytes = evicted, "Evicted buffers to make room");
                    }
                }

                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: min_size,
                    usage,
                    mapped_at_creation: false,
                });

                inner.total_allocated_bytes += min_size;
                inner.in_use_count += 1;

                tracing::debug!(
                    size = min_size,
                    label = label,
                    max_buffer_size,
                    total_mb = inner.total_allocated_bytes / (1024 * 1024),
                    "Allocated exact-size one-off buffer (no usable pool size class)"
                );

                PoolEntry {
                    buffer,
                    size_class: min_size,
                    usage,
                    last_used: Instant::now(),
                }
            }
            // Rejected before the lock was taken.
            AllocationPlan::TooLarge => return None,
        };

        drop(inner); // Release lock before creating PooledBuffer

        Some(PooledBuffer {
            entry: ManuallyDrop::new(entry),
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
    /// The pool entry (buffer + metadata). Always initialized for the
    /// lifetime of a live `PooledBuffer`; taken out exactly once, in
    /// `Drop::drop`, via `ManuallyDrop::take`. Using `ManuallyDrop` here
    /// (rather than `Option`) means there is no "already returned" state
    /// for safe code to ever observe, so `buffer()` and the `Deref` impl
    /// need no fallback branch and cannot panic.
    entry: ManuallyDrop<PoolEntry>,
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
        self.entry.size_class
    }

    /// Get the actual requested size.
    #[inline]
    #[must_use]
    pub fn actual_size(&self) -> u64 {
        self.actual_size
    }

    /// Get the underlying wgpu buffer.
    #[inline]
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.entry.buffer
    }

    /// Get the underlying wgpu buffer.
    ///
    /// Kept for API compatibility with the previous `Option`-based
    /// representation. A live `PooledBuffer` always holds its buffer (it is
    /// only released in `Drop`), so this now always returns `Some`; prefer
    /// [`Self::buffer`] in new code.
    #[inline]
    #[must_use]
    pub fn try_buffer(&self) -> Option<&wgpu::Buffer> {
        Some(&self.entry.buffer)
    }
}

impl Deref for PooledBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.entry.buffer
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Safety: `Drop::drop` runs at most once per value (Rust's
        // ownership model guarantees this), so this is the only place
        // `entry` is ever taken out of its `ManuallyDrop` wrapper, and no
        // other code can observe `self` afterward.
        let mut entry = unsafe { ManuallyDrop::take(&mut self.entry) };

        // Update last_used time for LRU
        entry.last_used = Instant::now();

        // Try to return to pool. A poisoned mutex is recovered rather than
        // silently skipped, so pool accounting (`in_use_count`,
        // `total_allocated_bytes`) cannot drift out from under a panic
        // elsewhere -- see `lock_recover`.
        if let Some(pool) = self.pool.upgrade() {
            lock_recover(&pool).return_buffer(entry);
            return;
        }

        // Pool itself is gone: nothing left to account for, buffer is
        // simply dropped.
        tracing::trace!(
            size_class = entry.size_class,
            "Buffer dropped (pool unavailable)"
        );
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

        // Regression: sizes just above the old 16MB ceiling used to return
        // `None` (the pool was inert for any readback above 1024x1024).
        // They now land in the 64MB class. A 1920x1080 f32 RGBA readback is
        // ~33MB and a 2560x1440 one is ~59MB -- both are covered now.
        assert_eq!(
            BufferPoolInner::size_class_index(16 * 1024 * 1024 + 1),
            Some(8)
        ); // > 16MB -> 64MB class
        assert_eq!(BufferPoolInner::size_class_index(33 * 1024 * 1024), Some(8)); // 1920x1080 RGBA f32 readback -> 64MB
        assert_eq!(BufferPoolInner::size_class_index(59 * 1024 * 1024), Some(8)); // 2560x1440 RGBA f32 readback -> 64MB
        assert_eq!(
            BufferPoolInner::size_class_index(256 * 1024 * 1024),
            Some(9)
        ); // 256MB -> 256MB
        assert_eq!(
            BufferPoolInner::size_class_index(256 * 1024 * 1024 + 1),
            None
        ); // > 256MB: no class covers it (served as a one-off, see `BufferPool::acquire`)
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
        assert_eq!(SIZE_CLASSES.len(), 10);
        assert_eq!(SIZE_CLASSES[0], 1024); // 1KB
        assert_eq!(SIZE_CLASSES[1], 4 * 1024); // 4KB
        assert_eq!(SIZE_CLASSES[2], 16 * 1024); // 16KB
        assert_eq!(SIZE_CLASSES[3], 64 * 1024); // 64KB
        assert_eq!(SIZE_CLASSES[4], 256 * 1024); // 256KB
        assert_eq!(SIZE_CLASSES[5], 1024 * 1024); // 1MB
        assert_eq!(SIZE_CLASSES[6], 4 * 1024 * 1024); // 4MB
        assert_eq!(SIZE_CLASSES[7], 16 * 1024 * 1024); // 16MB
        assert_eq!(SIZE_CLASSES[8], 64 * 1024 * 1024); // 64MB
        assert_eq!(SIZE_CLASSES[9], 256 * 1024 * 1024); // 256MB
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

    /// Regression test for two bugs together:
    ///
    /// 1. A cache hit used to reuse a pooled buffer regardless of its usage
    ///    flags, so a `MAP_READ`-flavoured buffer could be handed back for a
    ///    `STORAGE` request of the same size class (fails wgpu validation).
    ///    A cache hit must now only reuse an entry whose usage is a
    ///    superset of the request.
    /// 2. Requests above the largest size class used to return `None`
    ///    instead of being served; they must now succeed via a one-off
    ///    allocation.
    #[test]
    fn test_acquire_respects_usage_flags_and_oversized_fallback() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }));
        let adapter = match adapter {
            Ok(a) => a,
            Err(_) => {
                eprintln!("No GPU adapter available, skipping GPU test");
                return;
            }
        };

        let device = match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pool_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })) {
            Ok((device, _queue)) => device,
            Err(_) => {
                eprintln!("Failed to create GPU device, skipping GPU test");
                return;
            }
        };

        let pool = BufferPool::new(512 * 1024 * 1024);

        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let small_size = SIZE_CLASSES[0];

        // Acquire and drop a STORAGE buffer so it returns to the pool.
        drop(
            pool.acquire(&device, small_size, storage_usage, "storage")
                .expect("acquire should succeed"),
        );
        assert_eq!(
            pool.stats().available_count,
            1,
            "buffer should return to the pool on drop"
        );

        // A same-size request for a non-superset usage must NOT reuse the
        // STORAGE buffer -- it should allocate fresh instead.
        let staged = pool
            .acquire(&device, small_size, staging_usage, "staging")
            .expect("acquire should succeed");
        assert_eq!(
            staged.buffer().usage(),
            staging_usage,
            "must not reuse a buffer whose usage flags don't cover the request"
        );
        assert_eq!(
            pool.stats().available_count,
            1,
            "the original STORAGE buffer must remain untouched in the pool"
        );
        drop(staged);

        // A request whose usage the pooled STORAGE buffer covers should now
        // hit the cache (no fresh allocation).
        let allocations_before = pool.stats().total_allocations;
        let reused = pool
            .acquire(&device, small_size, storage_usage, "storage-reuse")
            .expect("acquire should succeed");
        assert_eq!(reused.buffer().usage(), storage_usage);
        assert_eq!(
            pool.stats().total_allocations,
            allocations_before,
            "matching usage should be served from the pool, not a fresh allocation"
        );
        drop(reused);

        // A request the *device* cannot satisfy must be declined gracefully.
        // Creating the buffer anyway is a wgpu validation error, which is
        // delivered to the device's uncaptured-error handler and aborts the
        // process -- it is not an error `acquire` could propagate, so the
        // limit has to be checked before allocating.
        let max_buffer_size = device.limits().max_buffer_size;
        assert!(
            pool.acquire(
                &device,
                max_buffer_size + 1,
                wgpu::BufferUsages::COPY_DST,
                "beyond-device-limit",
            )
            .is_none(),
            "a request above the device's max_buffer_size ({max_buffer_size}) must return None"
        );

        // Oversized fallback: a request larger than the largest size class
        // must still succeed (previously returned None and the pool was
        // inert for every readback above 1024x1024) -- as long as the device
        // can hold such a buffer at all. `wgpu::Limits::default()` caps
        // `max_buffer_size` at exactly the largest size class (256 MB), so on
        // a default-limits device this branch is unreachable and the
        // `TooLarge` assertion above is the meaningful one; the
        // `plan_allocation` unit tests cover both policies without a GPU.
        let oversized_size = SIZE_CLASSES[SIZE_CLASSES.len() - 1] + 1;
        if oversized_size <= max_buffer_size {
            let oversized = pool
                .acquire(
                    &device,
                    oversized_size,
                    wgpu::BufferUsages::COPY_DST,
                    "oversized",
                )
                .expect("requests above the largest size class must still be served");
            assert_eq!(oversized.actual_size(), oversized_size);
            assert!(oversized.buffer().size() >= oversized_size);
            drop(oversized);
        }
    }

    /// The device-limit policy behind `BufferPool::acquire`, unit-tested
    /// without a GPU (the interesting cases need a device whose
    /// `max_buffer_size` is not one of the size-class values).
    #[test]
    fn test_plan_allocation_respects_device_limit() {
        let largest = SIZE_CLASSES[SIZE_CLASSES.len() - 1];
        let generous = largest * 4;

        // Normal case: the size class fits the device.
        assert_eq!(
            BufferPoolInner::plan_allocation(1, generous),
            AllocationPlan::SizeClass(0)
        );
        assert_eq!(
            BufferPoolInner::plan_allocation(33 * 1024 * 1024, generous),
            AllocationPlan::SizeClass(8)
        );

        // Above the largest size class but within the device limit: exact
        // one-off allocation.
        assert_eq!(
            BufferPoolInner::plan_allocation(largest + 1, generous),
            AllocationPlan::Exact
        );

        // Above the device limit: unservable.
        assert_eq!(
            BufferPoolInner::plan_allocation(generous + 1, generous),
            AllocationPlan::TooLarge
        );
        assert_eq!(
            BufferPoolInner::plan_allocation(largest + 1, largest),
            AllocationPlan::TooLarge
        );

        // Request fits the device but its size class does not: the class is
        // downgraded to an exact-size allocation instead of refusing (or,
        // as before this policy existed, tripping wgpu validation).
        // Device limit 24 MB: a 20 MB request rounds up to the 64 MB class.
        let device_limit = 24 * 1024 * 1024;
        let request = 20 * 1024 * 1024;
        assert_eq!(BufferPoolInner::size_class_index(request), Some(8));
        assert!(SIZE_CLASSES[8] > device_limit);
        assert_eq!(
            BufferPoolInner::plan_allocation(request, device_limit),
            AllocationPlan::Exact
        );

        // Exactly at the limit is still servable.
        assert_eq!(
            BufferPoolInner::plan_allocation(largest, largest),
            AllocationPlan::SizeClass(SIZE_CLASSES.len() - 1)
        );
    }

    /// An exact-size one-off allocation must never be mistaken for a pooled
    /// size class on return (`return_buffer` identifies classes by value).
    #[test]
    fn test_exact_plan_sizes_never_collide_with_a_size_class() {
        // `plan_allocation` only returns `Exact` for sizes that are not
        // themselves size-class values: a size equal to `SIZE_CLASSES[i]`
        // always selects class `i`, and that class is <= the device limit
        // whenever the request is.
        for (idx, &class) in SIZE_CLASSES.iter().enumerate() {
            assert_eq!(
                BufferPoolInner::plan_allocation(class, class),
                AllocationPlan::SizeClass(idx),
                "a request of exactly SIZE_CLASSES[{idx}] must use that class"
            );
        }
    }
}
