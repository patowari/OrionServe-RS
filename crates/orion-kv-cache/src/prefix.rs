//! Automatic prefix caching.
//!
//! When many requests share a leading prompt — a system prompt, a few-shot
//! preamble, a long document being asked several questions — recomputing that
//! prefix's KV entries per request is pure waste. This index lets a new
//! sequence adopt already-computed blocks instead.
//!
//! # Correctness
//!
//! Reusing a block's KV values is only sound if those values would have been
//! recomputed identically. That holds when the entire preceding token history
//! matches, which is exactly what the chained hash from
//! [`hash_block`](crate::table::hash_block) encodes. The index additionally
//! stores the tokens of each cached block and compares them on lookup, so a
//! hash collision degrades to a miss rather than to silent corruption.

use std::collections::HashMap;

use orion_core::{BlockId, TokenId};

use crate::block::BlockHash;

/// What a cached block covers.
#[derive(Debug, Clone)]
struct CacheEntry {
    block: BlockId,
    /// The exact tokens this block holds, kept for collision verification.
    tokens: Vec<TokenId>,
}

/// Hash-indexed map from computed prefixes to physical blocks.
///
/// The index holds *no* references of its own: entries point at blocks whose
/// refcount may be zero. A zero-refcount cached block is still in the pool's
/// free list and may be recycled at any time, at which point
/// [`PrefixCache::remove`] must be called to drop the stale entry. Keeping the
/// index reference-free is what stops the cache pinning memory and starving
/// live requests.
#[derive(Debug, Default)]
pub struct PrefixCache {
    entries: HashMap<BlockHash, CacheEntry>,
    hits: u64,
    misses: u64,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Cache hit rate in `0.0..=1.0`; `0.0` before any lookup.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Looks up a block by chained hash, verifying token equality.
    ///
    /// Records a hit or miss. Returns the physical block to adopt; the caller
    /// must take a reference to it via the pool before using it.
    pub fn lookup(&mut self, hash: BlockHash, tokens: &[TokenId]) -> Option<BlockId> {
        match self.entries.get(&hash) {
            Some(entry) if entry.tokens == tokens => {
                self.hits += 1;
                Some(entry.block)
            }
            Some(_) => {
                // Hash collision: distinct token sequences reached the same
                // digest. Treat as a miss rather than returning wrong KV data.
                self.misses += 1;
                None
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Looks up without recording statistics, for tests and introspection.
    pub fn peek(&self, hash: BlockHash) -> Option<BlockId> {
        self.entries.get(&hash).map(|e| e.block)
    }

    /// Publishes a finished block under its hash.
    ///
    /// Only full blocks should be inserted: a partial block will still be
    /// written to, and sharing it would let one sequence observe another's
    /// tokens. Re-inserting an existing hash keeps the first block, since both
    /// hold identical data and churning the mapping would only invalidate
    /// references others may already have adopted.
    pub fn insert(&mut self, hash: BlockHash, block: BlockId, tokens: Vec<TokenId>) {
        self.entries
            .entry(hash)
            .or_insert(CacheEntry { block, tokens });
    }

    /// Drops the entry for `hash`, if the block it names is still `block`.
    ///
    /// The identity check matters: a block may have been evicted and its hash
    /// re-inserted by a different block in between, and blindly removing would
    /// then discard a live, valid entry.
    pub fn remove(&mut self, hash: BlockHash, block: BlockId) {
        if let Some(entry) = self.entries.get(&hash) {
            if entry.block == block {
                self.entries.remove(&hash);
            }
        }
    }

    /// Empties the index. Does not touch the pool.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Resets hit/miss counters without discarding cached blocks.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::hash_block;

    #[test]
    fn a_lookup_on_an_empty_cache_misses() {
        let mut cache = PrefixCache::new();
        assert!(cache.lookup(BlockHash(1), &[1, 2]).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn an_inserted_block_is_found_again() {
        let mut cache = PrefixCache::new();
        let h = hash_block(None, &[1, 2, 3]);
        cache.insert(h, BlockId(5), vec![1, 2, 3]);

        assert_eq!(cache.lookup(h, &[1, 2, 3]), Some(BlockId(5)));
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.hit_rate(), 1.0);
    }

    #[test]
    fn a_hash_collision_with_different_tokens_degrades_to_a_miss() {
        let mut cache = PrefixCache::new();
        let h = BlockHash(0xdead);
        cache.insert(h, BlockId(1), vec![1, 2, 3]);

        // Same digest, different tokens: must not hand back the wrong block.
        assert_eq!(cache.lookup(h, &[9, 9, 9]), None);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn reinserting_a_hash_keeps_the_original_block() {
        let mut cache = PrefixCache::new();
        let h = hash_block(None, &[1]);
        cache.insert(h, BlockId(1), vec![1]);
        cache.insert(h, BlockId(2), vec![1]);
        assert_eq!(cache.peek(h), Some(BlockId(1)));
    }

    #[test]
    fn removal_is_guarded_by_block_identity() {
        let mut cache = PrefixCache::new();
        let h = hash_block(None, &[1]);
        cache.insert(h, BlockId(1), vec![1]);

        // A stale removal naming a different block must not evict the entry.
        cache.remove(h, BlockId(99));
        assert_eq!(cache.peek(h), Some(BlockId(1)));

        cache.remove(h, BlockId(1));
        assert_eq!(cache.peek(h), None);
    }

    #[test]
    fn a_shared_prefix_hits_across_two_sequences() {
        // Two requests share "system prompt" tokens then diverge.
        let mut cache = PrefixCache::new();
        let shared = [10, 11, 12, 13];

        let h0 = hash_block(None, &shared);
        cache.insert(h0, BlockId(0), shared.to_vec());

        // Second request recomputes the same chained hash and hits.
        let h0_again = hash_block(None, &shared);
        assert_eq!(cache.lookup(h0_again, &shared), Some(BlockId(0)));

        // Their divergent second blocks do not collide.
        let a1 = hash_block(Some(h0), &[20, 21]);
        let b1 = hash_block(Some(h0), &[30, 31]);
        assert_ne!(a1, b1);
        assert!(cache.lookup(b1, &[30, 31]).is_none());
    }

    #[test]
    fn hit_rate_reflects_the_mix_of_lookups() {
        let mut cache = PrefixCache::new();
        let h = hash_block(None, &[1]);
        cache.insert(h, BlockId(0), vec![1]);

        cache.lookup(h, &[1]);
        cache.lookup(BlockHash(999), &[2]);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hit_rate(), 0.5);

        cache.reset_stats();
        assert_eq!(cache.hit_rate(), 0.0);
        assert_eq!(cache.len(), 1, "stats reset must not drop entries");
    }
}
