//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::FileEntry;

/// Split result for [`apply_split`]: `(train, val, test)` slices of [`FileEntry`] references.
pub(super) type SplitResult<'a> = (Vec<&'a FileEntry>, Vec<&'a FileEntry>, Vec<&'a FileEntry>);
