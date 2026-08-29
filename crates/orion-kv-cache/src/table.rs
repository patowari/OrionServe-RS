//! Per-sequence block tables and prefix hashing.
//!
//! A block table is the indirection that makes paging work: a sequence sees a
//! contiguous logical token space, while the physical blocks backing it may be
//! scattered anywhere in the pool. This is the same idea as virtual memory, and
//! it buys the same thing — no external fragmentation, and no need to reserve a
//! sequence's worst-case length up front.

use orion_core::{BlockId, TokenId};

use crate::block::BlockHash;

/// Maps a sequence's logical block indices to physical blocks.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Physical blocks in logical order. Index `i` covers logical tokens
    /// `i * block_size .. (i + 1) * block_size`.
    blocks: Vec<BlockId>,
    /// Tokens currently backed by these blocks. Distinct from
    /// `blocks.len() * block_size` because the last block is usually partial.
    num_tokens: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Appends a physical block to the end of the table.
    pub fn push(&mut self, id: BlockId) {
        self.blocks.push(id);
    }

    /// Sets the number of logically valid tokens.
    pub fn set_num_tokens(&mut self, n: usize) {
        self.num_tokens = n;
    }

    /// Free slots in the final block, or `0` when the table is empty or exactly
    /// full.
    pub fn slack(&self, block_size: usize) -> usize {
        let capacity = self.blocks.len() * block_size;
        capacity.saturating_sub(self.num_tokens)
    }

    /// Blocks needed to hold `num_tokens` in total.
    pub fn blocks_required(num_tokens: usize, block_size: usize) -> usize {
        num_tokens.div_ceil(block_size)
    }

    /// Additional blocks needed to grow to `target_tokens`.
    pub fn additional_blocks_for(&self, target_tokens: usize, block_size: usize) -> usize {
        Self::blocks_required(target_tokens, block_size).saturating_sub(self.blocks.len())
    }

    /// The physical block ids as `u32`, the layout the model backend consumes.
    pub fn to_raw(&self) -> Vec<u32> {
        self.blocks.iter().map(|b| b.0).collect()
    }

    /// Removes and returns every block, leaving the table empty.
    ///
    /// Used on preemption and cleanup; the caller is responsible for releasing
    /// the returned references to the pool.
    pub fn take_blocks(&mut self) -> Vec<BlockId> {
        self.num_tokens = 0;
        std::mem::take(&mut self.blocks)
    }

    /// Drops the trailing `n` blocks, returning them for release.
    pub fn truncate_blocks(&mut self, n: usize) -> Vec<BlockId> {
        let keep = self.blocks.len().saturating_sub(n);
        self.blocks.split_off(keep)
    }
}

/// Computes the chained content hash of one block of tokens.
///
/// `parent` is the hash of the preceding block, or `None` for the first block
/// of a sequence. Chaining is what makes the hash identify a *prefix* rather
/// than a bag of tokens: block `n` matches only if blocks `0..=n` all match, so
/// a cache hit guarantees the KV values were computed under identical context.
///
/// The mixing function is FxHash-style — a multiply and a rotate per word.
/// This is not a cryptographic hash and does not need to be: a collision would
/// have to be engineered by a client who can already see the model's outputs,
/// and [`PrefixCache`](crate::prefix::PrefixCache) verifies token equality on
/// every hit rather than trusting the hash alone.
pub fn hash_block(parent: Option<BlockHash>, tokens: &[TokenId]) -> BlockHash {
    let mut state = parent.map_or(0xcbf2_9ce4_8422_2325, |h| h.0);
    // Length is mixed in so that a short final block cannot collide with a
    // longer one sharing its leading tokens.
    state = mix(state, tokens.len() as u64);
    for &t in tokens {
        state = mix(state, t as u64);
    }
    BlockHash(state)
}

#[inline]
fn mix(state: u64, value: u64) -> u64 {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    (state.rotate_left(5) ^ value).wrapping_mul(SEED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_required_rounds_up() {
        assert_eq!(BlockTable::blocks_required(0, 16), 0);
        assert_eq!(BlockTable::blocks_required(1, 16), 1);
        assert_eq!(BlockTable::blocks_required(16, 16), 1);
        assert_eq!(BlockTable::blocks_required(17, 16), 2);
        assert_eq!(BlockTable::blocks_required(32, 16), 2);
    }

    #[test]
    fn slack_reports_room_in_the_final_block() {
        let mut t = BlockTable::new();
        assert_eq!(t.slack(16), 0, "empty table has no allocated slack");

        t.push(BlockId(0));
        t.set_num_tokens(10);
        assert_eq!(t.slack(16), 6);

        t.set_num_tokens(16);
        assert_eq!(t.slack(16), 0, "exactly full");
    }

    #[test]
    fn additional_blocks_accounts_for_what_is_already_held() {
        let mut t = BlockTable::new();
        t.push(BlockId(0));
        t.set_num_tokens(16);
        // Growing 16 -> 17 tokens needs one more block.
        assert_eq!(t.additional_blocks_for(17, 16), 1);
        // Growing within the existing block needs none.
        assert_eq!(t.additional_blocks_for(16, 16), 0);
        // Shrinking never asks for blocks.
        assert_eq!(t.additional_blocks_for(4, 16), 0);
    }

    #[test]
    fn take_blocks_empties_the_table() {
        let mut t = BlockTable::new();
        t.push(BlockId(1));
        t.push(BlockId(2));
        t.set_num_tokens(20);

        let taken = t.take_blocks();
        assert_eq!(taken, vec![BlockId(1), BlockId(2)]);
        assert!(t.is_empty());
        assert_eq!(t.num_tokens(), 0);
    }

    #[test]
    fn truncate_returns_only_the_trailing_blocks() {
        let mut t = BlockTable::new();
        for i in 0..4 {
            t.push(BlockId(i));
        }
        let dropped = t.truncate_blocks(2);
        assert_eq!(dropped, vec![BlockId(2), BlockId(3)]);
        assert_eq!(t.blocks(), &[BlockId(0), BlockId(1)]);
    }

    #[test]
    fn raw_conversion_preserves_logical_order() {
        let mut t = BlockTable::new();
        t.push(BlockId(9));
        t.push(BlockId(3));
        assert_eq!(t.to_raw(), vec![9, 3]);
    }

    #[test]
    fn identical_prefixes_hash_identically() {
        let a = hash_block(None, &[1, 2, 3]);
        let b = hash_block(None, &[1, 2, 3]);
        assert_eq!(a, b);

        let a2 = hash_block(Some(a), &[4, 5]);
        let b2 = hash_block(Some(b), &[4, 5]);
        assert_eq!(a2, b2);
    }

    #[test]
    fn different_tokens_hash_differently() {
        assert_ne!(hash_block(None, &[1, 2, 3]), hash_block(None, &[1, 2, 4]));
        assert_ne!(hash_block(None, &[1, 2, 3]), hash_block(None, &[3, 2, 1]));
    }

    #[test]
    fn the_same_tokens_under_a_different_prefix_hash_differently() {
        // This is the property that makes position-dependent KV reuse sound:
        // the same block content preceded by different history must not match.
        let p1 = hash_block(None, &[10, 11]);
        let p2 = hash_block(None, &[20, 21]);
        assert_ne!(hash_block(Some(p1), &[1, 2]), hash_block(Some(p2), &[1, 2]));
    }

    #[test]
    fn a_root_block_differs_from_the_same_tokens_mid_sequence() {
        let root = hash_block(None, &[1, 2]);
        assert_ne!(root, hash_block(Some(BlockHash(0)), &[1, 2]));
    }

    #[test]
    fn length_is_mixed_in_so_short_blocks_do_not_collide() {
        assert_ne!(hash_block(None, &[1, 2]), hash_block(None, &[1, 2, 0]));
    }
}
