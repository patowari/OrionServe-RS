//! The KV cache manager: allocation, growth, sharing and release.
//!
//! This is the component that owns all KV memory accounting. The scheduler
//! asks it whether a sequence can be admitted or extended, and it answers from
//! exact block counts rather than estimates — an inference engine that
//! over-commits its cache does not degrade gracefully, it fails a request
//! mid-generation.
//!
//! # Concurrency
//!
//! The manager is deliberately **not** internally synchronized. It is owned by
//! the single engine step loop, which is the only thing that may allocate or
//! free blocks, and it is mutated through `&mut self`. Wrapping it in a mutex
//! would invite callers to interleave allocation decisions with scheduling
//! decisions, which is precisely the race that produces double-allocation
//! bugs. Sharing, if it is ever needed, belongs at the engine level.

use std::collections::HashMap;

use orion_core::{EngineError, SequenceId, TokenId};

use crate::block::{BlockHash, BlockPool};
use crate::prefix::PrefixCache;
use crate::table::{hash_block, BlockTable};

/// Outcome of allocating a sequence's prompt blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationOutcome {
    /// Blocks newly allocated and computed by this sequence.
    pub allocated: usize,
    /// Blocks adopted from the prefix cache, whose KV entries already exist.
    pub reused: usize,
    /// Prompt tokens whose KV entries came from the cache and therefore need
    /// no prefill compute. Always a whole number of blocks.
    pub cached_tokens: usize,
}

impl AllocationOutcome {
    /// Whether the prefix cache contributed anything.
    pub fn had_cache_hit(&self) -> bool {
        self.reused > 0
    }
}

/// Snapshot of cache occupancy, for metrics and admission decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub free_blocks: usize,
    pub block_size: usize,
    pub num_sequences: usize,
    pub prefix_cache_entries: usize,
    pub prefix_cache_hits: u64,
    pub prefix_cache_misses: u64,
}

impl CacheStats {
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks as f64 / self.total_blocks as f64
        }
    }

    /// Tokens the cache could hold if perfectly packed.
    pub fn token_capacity(&self) -> usize {
        self.total_blocks * self.block_size
    }
}

/// Owns the block pool, every sequence's block table, and the prefix index.
#[derive(Debug)]
pub struct KvCacheManager {
    pool: BlockPool,
    tables: HashMap<SequenceId, BlockTable>,
    prefix_cache: PrefixCache,
    prefix_caching_enabled: bool,
    /// Trailing hash of each sequence's last full block, so appending a block
    /// can extend the chain without rehashing the whole prompt.
    chain_tips: HashMap<SequenceId, BlockHash>,
}

impl KvCacheManager {
    /// Creates a manager over `num_blocks` blocks of `block_size` tokens each.
    pub fn new(num_blocks: usize, block_size: usize, prefix_caching_enabled: bool) -> Self {
        Self {
            pool: BlockPool::new(num_blocks, block_size),
            tables: HashMap::new(),
            prefix_cache: PrefixCache::new(),
            prefix_caching_enabled,
            chain_tips: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.pool.block_size()
    }

    pub fn total_blocks(&self) -> usize {
        self.pool.total_blocks()
    }

    pub fn free_blocks(&self) -> usize {
        self.pool.num_free()
    }

    pub fn utilization(&self) -> f64 {
        self.pool.utilization()
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_blocks: self.pool.total_blocks(),
            used_blocks: self.pool.num_used(),
            free_blocks: self.pool.num_free(),
            block_size: self.pool.block_size(),
            num_sequences: self.tables.len(),
            prefix_cache_entries: self.prefix_cache.len(),
            prefix_cache_hits: self.prefix_cache.hits(),
            prefix_cache_misses: self.prefix_cache.misses(),
        }
    }

    pub fn prefix_cache_hit_rate(&self) -> f64 {
        self.prefix_cache.hit_rate()
    }

    /// The block table for a sequence, if it has one.
    pub fn block_table(&self, seq: SequenceId) -> Option<&BlockTable> {
        self.tables.get(&seq)
    }

    /// Whether a sequence currently holds cache blocks.
    pub fn is_allocated(&self, seq: SequenceId) -> bool {
        self.tables.contains_key(&seq)
    }

    /// Blocks needed to admit a prompt of `num_tokens`, ignoring cache reuse.
    ///
    /// The scheduler uses this as a conservative admission bound: reuse can
    /// only reduce the requirement, never increase it, so admitting on this
    /// figure is always safe.
    pub fn blocks_needed_for(&self, num_tokens: usize) -> usize {
        BlockTable::blocks_required(num_tokens, self.pool.block_size())
    }

    /// Whether `num_tokens` could be admitted right now.
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        self.blocks_needed_for(num_tokens) <= self.pool.num_free()
    }

    /// Allocates blocks covering a sequence's prompt, reusing cached prefix
    /// blocks where possible.
    ///
    /// # Failure atomicity
    ///
    /// If the pool runs out partway through, every block taken by this call is
    /// released before returning. A partially-allocated sequence would
    /// otherwise leak blocks that nothing owns, and the leak would be
    /// invisible until the pool was exhausted.
    pub fn allocate(
        &mut self,
        seq: SequenceId,
        prompt: &[TokenId],
    ) -> Result<AllocationOutcome, EngineError> {
        if self.tables.contains_key(&seq) {
            return Err(EngineError::Internal(format!(
                "{seq} already has a block table"
            )));
        }

        let block_size = self.pool.block_size();
        let needed = BlockTable::blocks_required(prompt.len(), block_size);
        if needed > self.pool.num_free() {
            return Err(EngineError::CacheExhausted {
                needed,
                available: self.pool.num_free(),
            });
        }

        let mut table = BlockTable::new();
        let mut outcome = AllocationOutcome {
            allocated: 0,
            reused: 0,
            cached_tokens: 0,
        };
        let mut chain: Option<BlockHash> = None;
        // Once one block misses, every later block's chained hash is unknown to
        // the cache too, so lookups stop rather than wasting hash work.
        let mut still_matching = self.prefix_caching_enabled;

        for chunk in prompt.chunks(block_size) {
            let is_full = chunk.len() == block_size;
            let hash = if still_matching && is_full {
                Some(hash_block(chain, chunk))
            } else {
                None
            };

            let adopted = match hash {
                Some(h) => self.prefix_cache.lookup(h, chunk).and_then(|blk| {
                    // add_ref can only fail on an unknown id, which would be a
                    // stale index entry; fall through to a fresh allocation.
                    self.pool.add_ref(blk).ok().map(|()| blk)
                }),
                None => None,
            };

            match adopted {
                Some(blk) => {
                    table.push(blk);
                    outcome.reused += 1;
                    outcome.cached_tokens += chunk.len();
                    chain = hash;
                }
                None => {
                    still_matching = false;
                    match self.pool.allocate() {
                        Ok((blk, evicted)) => {
                            // Recycling a cached block invalidates its index
                            // entry; drop it before anything can look it up.
                            if let Some(h) = evicted {
                                self.prefix_cache.remove(h, blk);
                            }
                            table.push(blk);
                            outcome.allocated += 1;
                            // Contents are recorded now; the hash is published
                            // only after the block is actually computed, by
                            // `commit_prefill`.
                            let _ = self.pool.set_contents(blk, chunk.len(), None);
                            chain = None;
                        }
                        Err(e) => {
                            // Roll back everything this call took.
                            for blk in table.take_blocks() {
                                let _ = self.pool.release(blk);
                            }
                            return Err(e);
                        }
                    }
                }
            }
        }

        table.set_num_tokens(prompt.len());
        self.tables.insert(seq, table);
        Ok(outcome)
    }

    /// Publishes the sequence's full prompt blocks to the prefix cache.
    ///
    /// Called once prefill has actually computed the KV entries — publishing
    /// earlier would let another sequence adopt blocks holding uninitialized
    /// memory. The trailing partial block is never published, since it will
    /// still be written to during decode.
    pub fn commit_prefill(&mut self, seq: SequenceId, prompt: &[TokenId]) {
        if !self.prefix_caching_enabled {
            return;
        }
        let block_size = self.pool.block_size();
        let Some(table) = self.tables.get(&seq) else {
            return;
        };

        let mut chain: Option<BlockHash> = None;
        for (i, chunk) in prompt.chunks(block_size).enumerate() {
            if chunk.len() != block_size {
                break; // partial tail: not shareable
            }
            let Some(&blk) = table.blocks().get(i) else {
                break;
            };
            let h = hash_block(chain, chunk);
            let _ = self.pool.set_contents(blk, chunk.len(), Some(h));
            self.prefix_cache.insert(h, blk, chunk.to_vec());
            chain = Some(h);
        }
        if let Some(tip) = chain {
            self.chain_tips.insert(seq, tip);
        }
    }

    /// Ensures a sequence has room for one more token, allocating if the
    /// current final block is full.
    ///
    /// This is the decode-step hot path: usually a bounds check and nothing
    /// more, since a block absorbs `block_size` tokens before needing another.
    pub fn append_token(&mut self, seq: SequenceId) -> Result<(), EngineError> {
        let block_size = self.pool.block_size();
        let table = self
            .tables
            .get(&seq)
            .ok_or_else(|| EngineError::Internal(format!("{seq} has no block table")))?;

        let new_len = table.num_tokens() + 1;
        let need_block = BlockTable::blocks_required(new_len, block_size) > table.num_blocks();

        if need_block {
            // Allocate before mutating the table so a failure leaves the
            // sequence exactly as it was.
            let (blk, evicted) = self
                .pool
                .allocate()
                .map_err(|_| EngineError::CacheExhausted {
                    needed: 1,
                    available: self.pool.num_free(),
                })?;
            if let Some(h) = evicted {
                self.prefix_cache.remove(h, blk);
            }
            let table = self
                .tables
                .get_mut(&seq)
                .ok_or_else(|| EngineError::Internal(format!("{seq} vanished during append")))?;
            table.push(blk);
        }

        let table = self
            .tables
            .get_mut(&seq)
            .ok_or_else(|| EngineError::Internal(format!("{seq} vanished during append")))?;
        table.set_num_tokens(new_len);

        // Keep the final block's token count accurate so accounting stays
        // truthful even mid-block.
        let last = *table
            .blocks()
            .last()
            .ok_or_else(|| EngineError::Internal(format!("{seq} has an empty block table")))?;
        let filled = new_len - (table.num_blocks() - 1) * block_size;
        self.pool.set_contents(last, filled, None)?;
        Ok(())
    }

    /// Releases every block held by a sequence.
    ///
    /// Idempotent: releasing an unknown sequence is a no-op, so cleanup paths
    /// can run unconditionally without racing the normal completion path.
    pub fn free(&mut self, seq: SequenceId) {
        self.chain_tips.remove(&seq);
        let Some(mut table) = self.tables.remove(&seq) else {
            return;
        };
        for blk in table.take_blocks() {
            match self.pool.release(blk) {
                Ok(_) => {}
                Err(e) => {
                    // A release failure means the accounting is already
                    // corrupt. Log rather than panic: dropping one request's
                    // cleanup must not take down the engine.
                    tracing::error!(sequence = %seq, block = %blk, error = %e,
                        "failed to release KV block");
                }
            }
        }
    }

    /// Frees a sequence's blocks on preemption, returning how many were
    /// reclaimed.
    ///
    /// Identical to [`free`](Self::free) today. It exists as a distinct entry
    /// point because swap-based preemption will need to copy blocks out here
    /// rather than discard them, and the scheduler should not have to change
    /// when that lands.
    pub fn preempt(&mut self, seq: SequenceId) -> usize {
        let n = self
            .tables
            .get(&seq)
            .map(|t| t.num_blocks())
            .unwrap_or_default();
        self.free(seq);
        n
    }

    /// The raw block table a backend needs for this sequence.
    pub fn raw_block_table(&self, seq: SequenceId) -> Option<Vec<u32>> {
        self.tables.get(&seq).map(|t| t.to_raw())
    }

    /// Forgets cached contents until `needed` blocks are free of cached data,
    /// or until nothing more can be evicted.
    ///
    /// Returns the number of blocks evicted.
    ///
    /// Note what this does *not* do: eviction never changes `free_blocks`,
    /// because a cached block was already free and already allocatable. Its
    /// only effect is to drop prefix-cache entries, trading future hit rate for
    /// a smaller index. Blocks a sequence still references are never touched.
    ///
    /// Callers therefore do not need to evict in order to allocate — the pool
    /// recycles cached blocks on its own. This exists for bounding index size
    /// and for tests that need a cold cache.
    pub fn evict_cached(&mut self, count: usize) -> usize {
        let mut evicted = 0;
        while evicted < count {
            match self.pool.evict_oldest_cached() {
                Some((blk, hash)) => {
                    self.prefix_cache.remove(hash, blk);
                    evicted += 1;
                }
                None => break,
            }
        }
        evicted
    }

    /// Drops every prefix-cache entry, leaving the blocks themselves alone.
    pub fn clear_prefix_cache(&mut self) {
        let n = self.pool.num_evictable();
        self.evict_cached(n);
        self.prefix_cache.clear();
    }

    /// Free blocks that still carry reusable cached contents.
    pub fn num_evictable(&self) -> usize {
        self.pool.num_evictable()
    }

    /// Number of sequences currently holding blocks.
    pub fn num_sequences(&self) -> usize {
        self.tables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager(blocks: usize) -> KvCacheManager {
        KvCacheManager::new(blocks, 4, true)
    }

    fn seq_id(n: u64) -> SequenceId {
        SequenceId::from_raw(n)
    }

    #[test]
    fn allocation_reserves_the_right_number_of_blocks() {
        let mut m = manager(16);
        let s = seq_id(1);
        // 10 tokens at block_size 4 -> 3 blocks (4 + 4 + 2).
        let out = m.allocate(s, &[1; 10]).unwrap();
        assert_eq!(out.allocated, 3);
        assert_eq!(out.reused, 0);
        assert_eq!(m.free_blocks(), 13);
        assert_eq!(m.block_table(s).unwrap().num_tokens(), 10);
    }

    #[test]
    fn allocating_the_same_sequence_twice_is_an_internal_error() {
        let mut m = manager(16);
        let s = seq_id(1);
        m.allocate(s, &[1; 4]).unwrap();
        assert!(matches!(
            m.allocate(s, &[1; 4]).unwrap_err(),
            EngineError::Internal(_)
        ));
    }

    #[test]
    fn allocation_beyond_capacity_is_refused_before_taking_blocks() {
        let mut m = manager(2);
        let err = m.allocate(seq_id(1), &[1; 100]).unwrap_err();
        assert!(matches!(err, EngineError::CacheExhausted { .. }));
        assert_eq!(m.free_blocks(), 2, "refused allocation must not leak");
        assert!(!m.is_allocated(seq_id(1)));
    }

    #[test]
    fn freeing_returns_every_block() {
        let mut m = manager(8);
        let s = seq_id(1);
        m.allocate(s, &[1; 12]).unwrap();
        assert_eq!(m.free_blocks(), 5);
        m.free(s);
        assert_eq!(m.free_blocks(), 8);
        assert!(!m.is_allocated(s));
    }

    #[test]
    fn freeing_an_unknown_sequence_is_a_no_op() {
        let mut m = manager(4);
        m.free(seq_id(42));
        assert_eq!(m.free_blocks(), 4);
    }

    #[test]
    fn appending_reuses_the_final_block_until_it_fills() {
        let mut m = manager(8);
        let s = seq_id(1);
        // 4 tokens exactly fills one block.
        m.allocate(s, &[1; 4]).unwrap();
        assert_eq!(m.block_table(s).unwrap().num_blocks(), 1);
        assert_eq!(m.free_blocks(), 7);

        // Token 5 needs a second block.
        m.append_token(s).unwrap();
        assert_eq!(m.block_table(s).unwrap().num_blocks(), 2);
        assert_eq!(m.free_blocks(), 6);

        // Tokens 6-8 fit in the block just allocated.
        for _ in 0..3 {
            m.append_token(s).unwrap();
        }
        assert_eq!(m.block_table(s).unwrap().num_blocks(), 2);
        assert_eq!(m.block_table(s).unwrap().num_tokens(), 8);
        assert_eq!(m.free_blocks(), 6);
    }

    #[test]
    fn append_fails_cleanly_when_the_pool_is_exhausted() {
        let mut m = KvCacheManager::new(1, 4, false);
        let s = seq_id(1);
        m.allocate(s, &[1; 4]).unwrap();
        assert_eq!(m.free_blocks(), 0);

        let err = m.append_token(s).unwrap_err();
        assert!(matches!(err, EngineError::CacheExhausted { .. }));
        // The sequence is untouched and still usable.
        assert_eq!(m.block_table(s).unwrap().num_tokens(), 4);
        assert_eq!(m.block_table(s).unwrap().num_blocks(), 1);
    }

    #[test]
    fn append_to_an_unknown_sequence_is_an_internal_error() {
        let mut m = manager(4);
        assert!(matches!(
            m.append_token(seq_id(9)).unwrap_err(),
            EngineError::Internal(_)
        ));
    }

    #[test]
    fn a_shared_prefix_is_reused_by_a_second_sequence() {
        let mut m = manager(32);
        let prompt: Vec<TokenId> = (0..12).collect();

        let a = seq_id(1);
        let first = m.allocate(a, &prompt).unwrap();
        assert_eq!(first.allocated, 3);
        assert_eq!(first.reused, 0);
        m.commit_prefill(a, &prompt);

        // Second identical prompt should adopt all three full blocks.
        let b = seq_id(2);
        let second = m.allocate(b, &prompt).unwrap();
        assert_eq!(second.reused, 3, "all full blocks should be shared");
        assert_eq!(second.allocated, 0);
        assert_eq!(second.cached_tokens, 12);
        assert!(second.had_cache_hit());

        // Sharing means the second sequence consumed no new blocks.
        assert_eq!(m.free_blocks(), 29);
        assert_eq!(
            m.block_table(a).unwrap().blocks(),
            m.block_table(b).unwrap().blocks()
        );
    }

    #[test]
    fn shared_blocks_survive_until_the_last_sequence_releases_them() {
        let mut m = manager(32);
        let prompt: Vec<TokenId> = (0..8).collect();
        let (a, b) = (seq_id(1), seq_id(2));

        m.allocate(a, &prompt).unwrap();
        m.commit_prefill(a, &prompt);
        m.allocate(b, &prompt).unwrap();
        assert_eq!(m.free_blocks(), 30);

        m.free(a);
        assert_eq!(m.free_blocks(), 30, "b still references the blocks");
        m.free(b);
        assert_eq!(m.free_blocks(), 32);
    }

    #[test]
    fn a_divergent_prompt_shares_only_its_common_prefix() {
        let mut m = manager(32);
        let a_prompt: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // Same first block, different second.
        let b_prompt: Vec<TokenId> = vec![1, 2, 3, 4, 9, 9, 9, 9];

        let a = seq_id(1);
        m.allocate(a, &a_prompt).unwrap();
        m.commit_prefill(a, &a_prompt);

        let b = seq_id(2);
        let out = m.allocate(b, &b_prompt).unwrap();
        assert_eq!(out.reused, 1, "only the shared first block");
        assert_eq!(out.allocated, 1);
        assert_eq!(out.cached_tokens, 4);
    }

    #[test]
    fn a_partial_trailing_block_is_never_shared() {
        let mut m = manager(32);
        // 6 tokens: one full block plus a partial one.
        let prompt: Vec<TokenId> = (0..6).collect();
        let a = seq_id(1);
        m.allocate(a, &prompt).unwrap();
        m.commit_prefill(a, &prompt);

        let b = seq_id(2);
        let out = m.allocate(b, &prompt).unwrap();
        assert_eq!(out.reused, 1, "only the full block is shareable");
        assert_eq!(out.allocated, 1, "the partial tail is recomputed");
    }

    #[test]
    fn uncommitted_prefill_is_not_shared() {
        // Publishing before compute would hand out uninitialized KV entries.
        let mut m = manager(32);
        let prompt: Vec<TokenId> = (0..8).collect();
        m.allocate(seq_id(1), &prompt).unwrap();

        let out = m.allocate(seq_id(2), &prompt).unwrap();
        assert_eq!(out.reused, 0);
        assert_eq!(out.allocated, 2);
    }

    #[test]
    fn prefix_caching_can_be_disabled() {
        let mut m = KvCacheManager::new(32, 4, false);
        let prompt: Vec<TokenId> = (0..8).collect();
        let a = seq_id(1);
        m.allocate(a, &prompt).unwrap();
        m.commit_prefill(a, &prompt);

        let out = m.allocate(seq_id(2), &prompt).unwrap();
        assert_eq!(out.reused, 0);
        assert_eq!(out.allocated, 2);
    }

    #[test]
    fn preemption_reclaims_blocks_and_reports_the_count() {
        let mut m = manager(8);
        let s = seq_id(1);
        m.allocate(s, &[1; 12]).unwrap();
        assert_eq!(m.preempt(s), 3);
        assert_eq!(m.free_blocks(), 8);
        assert!(!m.is_allocated(s));
    }

    #[test]
    fn eviction_drops_index_entries_so_later_prompts_miss() {
        let mut m = manager(4);
        let prompt: Vec<TokenId> = (0..8).collect();
        let a = seq_id(1);
        m.allocate(a, &prompt).unwrap();
        m.commit_prefill(a, &prompt);
        m.free(a);

        // Blocks are free but still hold reusable contents.
        assert_eq!(m.stats().prefix_cache_entries, 2);
        assert_eq!(m.num_evictable(), 2);
        assert_eq!(m.free_blocks(), 4);

        assert_eq!(m.evict_cached(2), 2);
        assert_eq!(m.stats().prefix_cache_entries, 0);
        assert_eq!(
            m.free_blocks(),
            4,
            "eviction only forgets contents; the blocks were already free"
        );

        // A subsequent identical prompt therefore misses.
        let out = m.allocate(seq_id(2), &prompt).unwrap();
        assert_eq!(out.reused, 0);
    }

    #[test]
    fn eviction_stops_once_nothing_is_cached() {
        let mut m = manager(4);
        let prompt: Vec<TokenId> = (0..8).collect();
        let a = seq_id(1);
        m.allocate(a, &prompt).unwrap();
        m.commit_prefill(a, &prompt);
        m.free(a);

        // Asking for more than exists terminates rather than looping.
        assert_eq!(m.evict_cached(100), 2);
        assert_eq!(m.evict_cached(1), 0);
    }

    #[test]
    fn clearing_the_prefix_cache_leaves_blocks_allocatable() {
        let mut m = manager(4);
        let prompt: Vec<TokenId> = (0..8).collect();
        let a = seq_id(1);
        m.allocate(a, &prompt).unwrap();
        m.commit_prefill(a, &prompt);
        m.free(a);

        m.clear_prefix_cache();
        assert_eq!(m.stats().prefix_cache_entries, 0);
        assert_eq!(m.free_blocks(), 4);
        // The pool is still fully usable afterwards.
        m.allocate(seq_id(2), &prompt).unwrap();
        assert_eq!(m.free_blocks(), 2);
    }

    #[test]
    fn cached_blocks_are_recycled_under_pressure_without_explicit_eviction() {
        // The pool reclaims cached-but-free blocks on its own, so a full cache
        // never blocks a new allocation.
        let mut m = manager(2);
        let a_prompt: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let a = seq_id(1);
        m.allocate(a, &a_prompt).unwrap();
        m.commit_prefill(a, &a_prompt);
        m.free(a);
        assert_eq!(m.num_evictable(), 2);

        // A completely different prompt needs both blocks and must get them.
        let b_prompt: Vec<TokenId> = vec![9, 9, 9, 9, 9, 9, 9, 9];
        let out = m.allocate(seq_id(2), &b_prompt).unwrap();
        assert_eq!(out.allocated, 2);
        assert_eq!(out.reused, 0);
        assert_eq!(m.free_blocks(), 0);
    }

    #[test]
    fn recycling_a_cached_block_invalidates_its_index_entry() {
        // The dangerous case: a cached block is recycled for unrelated data
        // while its prefix-cache entry still names it. A later lookup for the
        // original prefix must miss, not adopt the overwritten block.
        let mut m = manager(2);
        let original: Vec<TokenId> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let a = seq_id(1);
        m.allocate(a, &original).unwrap();
        m.commit_prefill(a, &original);
        m.free(a);
        assert_eq!(m.stats().prefix_cache_entries, 2);

        // A different prompt takes both blocks, overwriting their contents.
        let other: Vec<TokenId> = vec![9; 8];
        let b = seq_id(2);
        m.allocate(b, &other).unwrap();
        assert_eq!(
            m.stats().prefix_cache_entries,
            0,
            "recycled blocks must not leave entries behind"
        );
        m.free(b);

        // The original prompt now genuinely misses and recomputes.
        let out = m.allocate(seq_id(3), &original).unwrap();
        assert_eq!(out.reused, 0, "must not adopt an overwritten block");
        assert_eq!(out.allocated, 2);
    }

    #[test]
    fn stats_report_capacity_and_utilization() {
        let mut m = manager(10);
        m.allocate(seq_id(1), &[1; 8]).unwrap();
        let s = m.stats();
        assert_eq!(s.total_blocks, 10);
        assert_eq!(s.used_blocks, 2);
        assert_eq!(s.free_blocks, 8);
        assert_eq!(s.num_sequences, 1);
        assert_eq!(s.token_capacity(), 40);
        assert_eq!(s.utilization(), 0.2);
    }

    #[test]
    fn admission_checks_agree_with_actual_allocation() {
        let m = manager(3);
        assert!(m.can_allocate(12));
        assert!(!m.can_allocate(13));
        assert_eq!(m.blocks_needed_for(13), 4);
    }

    #[test]
    fn an_empty_prompt_allocates_nothing() {
        let mut m = manager(4);
        let out = m.allocate(seq_id(1), &[]).unwrap();
        assert_eq!(out.allocated, 0);
        assert_eq!(m.free_blocks(), 4);
    }

    #[test]
    fn many_sequences_allocate_and_free_without_leaking() {
        let mut m = manager(64);
        for round in 0..10 {
            let ids: Vec<_> = (0..8).map(|i| seq_id(round * 100 + i)).collect();
            for &s in &ids {
                m.allocate(s, &[1; 16]).unwrap();
            }
            assert_eq!(m.free_blocks(), 64 - 8 * 4);
            for &s in &ids {
                m.free(s);
            }
            assert_eq!(m.free_blocks(), 64, "round {round} leaked blocks");
            assert_eq!(m.num_sequences(), 0);
        }
    }
}
