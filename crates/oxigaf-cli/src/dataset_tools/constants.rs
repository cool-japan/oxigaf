//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Maximum directory recursion depth for [`DatasetScanner::scan`].
///
/// A backstop against pathological/adversarial directory trees; the primary
/// cycle guard is the canonicalised visited-path set in `scan_dir`, which
/// alone is sufficient to stop a genuine symlink cycle.
pub(super) const MAX_SCAN_DEPTH: usize = 128;
