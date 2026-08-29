//! Block-based paged KV cache management.
//!
//! The KV cache is where an inference server's memory actually goes, and how
//! it is managed determines how many requests fit on a GPU. This crate
//! implements a paged cache: fixed-size blocks handed out from a pool, indexed
//! per sequence by a block table, with reference counting so sequences can
//! share identical prompt prefixes.
//!
//! See `docs/kv-cache.md` for the design rationale and the block-size analysis.
//!
//! # Module map
//!
//! * [`block`] — physical blocks, reference counts and the free pool.
//! * [`table`] — per-sequence logical-to-physical mapping, and prefix hashing.
//! * [`prefix`] — the hash index behind automatic prefix caching.
//! * [`manager`] — the composed public API used by the scheduler.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod block;
pub mod manager;
pub mod prefix;
pub mod table;

pub use block::{Block, BlockHash, BlockPool};
pub use manager::{AllocationOutcome, CacheStats, KvCacheManager};
pub use prefix::PrefixCache;
pub use table::{hash_block, BlockTable};
