//! Strongly-typed identifiers.
//!
//! These are newtypes rather than bare integers on purpose. A `BlockId` and a
//! `SequenceId` are both indices into different arenas, and the compiler should
//! reject swapping them — this class of bug is otherwise very hard to find in a
//! block-table implementation where both are `usize`.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Identifies a client request for the whole of its lifetime, including in
/// logs, traces and metrics exemplars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(u64);

impl RequestId {
    /// Allocates a process-unique id.
    ///
    /// A monotonic counter rather than a UUID: ids are compared and hashed on
    /// every scheduler pass, and a `u64` keeps `RequestId` `Copy` and cheap.
    /// The externally visible id in API responses is a separate string.
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        RequestId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Constructs an id from a raw value. For tests and deserialization only.
    pub const fn from_raw(v: u64) -> Self {
        RequestId(v)
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "req-{}", self.0)
    }
}

/// Identifies one decoding sequence.
///
/// Distinct from [`RequestId`] because a single request can fan out into
/// several sequences (beam search, or `n > 1` completions) that share a prompt
/// and therefore share KV blocks. The 1:1 case is just the common one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        SequenceId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub const fn from_raw(v: u64) -> Self {
        SequenceId(v)
    }
}

impl fmt::Display for SequenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq-{}", self.0)
    }
}

/// Index of a physical block in the KV cache pool.
///
/// "Physical" is the important word: a sequence addresses its cache through a
/// block *table* of these, and the ids need not be contiguous or ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blk-{}", self.0)
    }
}

/// A token id as produced by the tokenizer and consumed by the model.
pub type TokenId = u32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_ids_are_unique_and_monotonic() {
        let a = RequestId::next();
        let b = RequestId::next();
        assert!(b.as_u64() > a.as_u64());
        assert_ne!(a, b);
    }

    #[test]
    fn ids_display_with_a_readable_prefix() {
        assert_eq!(RequestId::from_raw(7).to_string(), "req-7");
        assert_eq!(SequenceId::from_raw(7).to_string(), "seq-7");
        assert_eq!(BlockId(7).to_string(), "blk-7");
    }
}
