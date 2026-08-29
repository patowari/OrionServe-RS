//! Test doubles for scheduler tests.
//!
//! Available under `cfg(test)` and behind the `testing` feature, so downstream
//! crates writing scheduler simulations can use the same fake.

use std::collections::HashMap;

use orion_core::{EngineError, KvCacheManagerLike, SequenceId, TokenId};

/// A cache that accounts blocks exactly like the real one but can be told to
/// fail on demand.
///
/// Forcing a real [`KvCacheManager`](orion_kv_cache::KvCacheManager) into
/// exhaustion requires arranging specific block arithmetic, which makes the
/// test about the arithmetic rather than about the scheduler's reaction. This
/// fake makes the failure itself the input.
#[derive(Debug)]
pub struct FakeCache {
    total: usize,
    block_size: usize,
    tables: HashMap<SequenceId, usize>,
    used: usize,
    fail_next_append: bool,
    fail_next_allocate: bool,
    pub commits: Vec<SequenceId>,
}

impl FakeCache {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            total: total_blocks,
            block_size,
            tables: HashMap::new(),
            used: 0,
            fail_next_append: false,
            fail_next_allocate: false,
            commits: Vec::new(),
        }
    }

    /// Makes the next [`append_token`](KvCacheManagerLike::append_token) fail
    /// with [`EngineError::CacheExhausted`].
    pub fn fail_next_append(&mut self) {
        self.fail_next_append = true;
    }

    /// Makes the next [`allocate`](KvCacheManagerLike::allocate) fail.
    pub fn fail_next_allocate(&mut self) {
        self.fail_next_allocate = true;
    }

    /// Blocks currently held by a sequence.
    pub fn blocks_held(&self, seq: SequenceId) -> usize {
        self.tables.get(&seq).copied().unwrap_or(0)
    }

    /// Number of sequences holding blocks.
    pub fn num_sequences(&self) -> usize {
        self.tables.len()
    }
}

impl KvCacheManagerLike for FakeCache {
    fn blocks_needed_for(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    fn total_blocks(&self) -> usize {
        self.total
    }

    fn free_blocks(&self) -> usize {
        self.total - self.used
    }

    fn can_allocate(&self, num_tokens: usize) -> bool {
        self.blocks_needed_for(num_tokens) <= self.free_blocks()
    }

    fn allocate(&mut self, seq: SequenceId, prompt: &[TokenId]) -> Result<(), EngineError> {
        if std::mem::take(&mut self.fail_next_allocate) {
            return Err(EngineError::CacheExhausted {
                needed: 1,
                available: 0,
            });
        }
        let needed = self.blocks_needed_for(prompt.len());
        if needed > self.free_blocks() {
            return Err(EngineError::CacheExhausted {
                needed,
                available: self.free_blocks(),
            });
        }
        self.used += needed;
        self.tables.insert(seq, needed);
        Ok(())
    }

    fn append_token(&mut self, seq: SequenceId) -> Result<(), EngineError> {
        if std::mem::take(&mut self.fail_next_append) {
            return Err(EngineError::CacheExhausted {
                needed: 1,
                available: 0,
            });
        }
        let held = self
            .tables
            .get(&seq)
            .copied()
            .ok_or_else(|| EngineError::Internal(format!("{seq} has no block table")))?;
        // Grow one block at a time, mirroring the real manager closely enough
        // that pressure behaves the same way.
        if self.free_blocks() == 0 {
            return Err(EngineError::CacheExhausted {
                needed: 1,
                available: 0,
            });
        }
        self.used += 1;
        self.tables.insert(seq, held + 1);
        Ok(())
    }

    fn free(&mut self, seq: SequenceId) {
        if let Some(n) = self.tables.remove(&seq) {
            self.used = self.used.saturating_sub(n);
        }
    }

    fn commit_prefill(&mut self, seq: SequenceId, _prompt: &[TokenId]) {
        self.commits.push(seq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fake_accounts_blocks_like_the_real_manager() {
        let mut c = FakeCache::new(8, 4);
        let s = SequenceId::from_raw(1);
        assert_eq!(c.blocks_needed_for(10), 3);
        c.allocate(s, &[1; 10]).unwrap();
        assert_eq!(c.free_blocks(), 5);
        c.free(s);
        assert_eq!(c.free_blocks(), 8);
    }

    #[test]
    fn forced_failures_fire_exactly_once() {
        let mut c = FakeCache::new(8, 4);
        let s = SequenceId::from_raw(1);
        c.allocate(s, &[1; 4]).unwrap();

        c.fail_next_append();
        assert!(c.append_token(s).is_err());
        assert!(c.append_token(s).is_ok(), "failure must not be sticky");
    }

    #[test]
    fn allocation_beyond_capacity_fails() {
        let mut c = FakeCache::new(2, 4);
        assert!(c.allocate(SequenceId::from_raw(1), &[1; 100]).is_err());
        assert_eq!(c.free_blocks(), 2, "a failed allocation must not consume");
    }
}
