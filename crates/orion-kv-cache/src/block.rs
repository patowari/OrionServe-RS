//! Physical blocks and the free pool that hands them out.
//!
//! A block is a fixed-size slice of the KV cache arena holding `block_size`
//! tokens' worth of keys and values for every layer. Blocks are addressed by
//! [`BlockId`] and are never moved, so a sequence's block table stays valid for
//! as long as it holds references.

use std::collections::VecDeque;

use orion_core::{BlockId, EngineError};

/// Content hash of a block's tokens, used to find shareable prefixes.
///
/// The hash chains the *previous* block's hash into the current one, so two
/// blocks collide only when their entire preceding token history matches. A
/// bare hash of the block's own 16 tokens would wrongly match identical text
/// appearing at different positions, whose KV values differ because attention
/// saw different context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockHash(pub u64);

/// State of one physical block.
#[derive(Debug, Clone)]
pub struct Block {
    id: BlockId,
    /// Number of sequences (and the prefix cache) currently referencing this
    /// block. A block is reclaimable exactly when this reaches zero.
    ref_count: u32,
    /// Tokens actually written, `0..=block_size`. Only a full block is
    /// eligible for prefix sharing, since a partial block will still be
    /// mutated.
    num_tokens: usize,
    /// Content hash, present only once the block is full and hashed.
    hash: Option<BlockHash>,
}

impl Block {
    fn new(id: BlockId) -> Self {
        Self {
            id,
            ref_count: 0,
            num_tokens: 0,
            hash: None,
        }
    }

    pub fn id(&self) -> BlockId {
        self.id
    }

    pub fn ref_count(&self) -> u32 {
        self.ref_count
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn hash(&self) -> Option<BlockHash> {
        self.hash
    }

    /// Whether this block is referenced by anything.
    pub fn is_free(&self) -> bool {
        self.ref_count == 0
    }

    /// Whether the block may be shared with another sequence: it must be full
    /// and hashed, so its contents can never change again.
    pub fn is_shareable(&self, block_size: usize) -> bool {
        self.hash.is_some() && self.num_tokens == block_size
    }

    /// Clears the block for reuse by a different sequence.
    fn reset(&mut self) {
        self.ref_count = 0;
        self.num_tokens = 0;
        self.hash = None;
    }
}

/// Owns every physical block and tracks which are free.
///
/// # Reclamation policy
///
/// Freed blocks go to the *back* of a FIFO queue rather than being reused
/// immediately. That is what makes prefix caching work without a separate
/// cache tier: a block whose refcount drops to zero keeps its contents and its
/// hash, so a later request with the same prefix can still reclaim it by hash.
/// Only when the pool is under pressure does the oldest such block get
/// recycled — an LRU eviction that falls out of the FIFO ordering for free.
#[derive(Debug)]
pub struct BlockPool {
    blocks: Vec<Block>,
    /// Ids with `ref_count == 0`, oldest first. Also the eviction order.
    free_list: VecDeque<BlockId>,
    block_size: usize,
}

impl BlockPool {
    /// Creates a pool of `num_blocks` blocks, each holding `block_size` tokens.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let blocks = (0..num_blocks)
            .map(|i| Block::new(BlockId(i as u32)))
            .collect();
        let free_list = (0..num_blocks).map(|i| BlockId(i as u32)).collect();
        Self {
            blocks,
            free_list,
            block_size,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Blocks with no live references. Some may still hold reusable contents.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    pub fn num_used(&self) -> usize {
        self.blocks.len() - self.free_list.len()
    }

    /// Fraction of the pool currently referenced, in `0.0..=1.0`.
    pub fn utilization(&self) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        self.num_used() as f64 / self.blocks.len() as f64
    }

    pub fn get(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(id.index())
    }

    /// Allocates a fresh block with `ref_count == 1` and no contents.
    ///
    /// Takes the oldest free block, which is the correct victim under the LRU
    /// ordering described on [`BlockPool`].
    ///
    /// Returns the block together with the content hash it *used* to hold, if
    /// any. The caller must drop that hash from any index pointing at this
    /// block: the contents are gone, and an index entry surviving the recycle
    /// would hand a later lookup a block full of unrelated data.
    pub fn allocate(&mut self) -> Result<(BlockId, Option<BlockHash>), EngineError> {
        let id = self
            .free_list
            .pop_front()
            .ok_or(EngineError::CacheExhausted {
                needed: 1,
                available: 0,
            })?;
        let block = &mut self.blocks[id.index()];
        debug_assert!(block.is_free(), "{id} was on the free list with live refs");
        let evicted = block.hash;
        block.reset();
        block.ref_count = 1;
        Ok((id, evicted))
    }

    /// Takes an additional reference to an already-live or cached block.
    ///
    /// Used when a sequence adopts a block from the prefix cache, and when a
    /// forked sequence shares its parent's prompt blocks.
    pub fn add_ref(&mut self, id: BlockId) -> Result<(), EngineError> {
        let block = self
            .blocks
            .get_mut(id.index())
            .ok_or_else(|| EngineError::Internal(format!("add_ref on unknown block {id}")))?;
        // Reviving a block that was sitting free: it leaves the free list but
        // keeps its contents, which is exactly the prefix-cache hit path.
        if block.ref_count == 0 {
            remove_from_free_list(&mut self.free_list, id);
        }
        block.ref_count += 1;
        Ok(())
    }

    /// Drops one reference. Returns `true` if the block became free.
    ///
    /// A freed block keeps its contents and hash so the prefix cache can still
    /// reclaim it; it is only cleared if and when it is reallocated.
    pub fn release(&mut self, id: BlockId) -> Result<bool, EngineError> {
        let block = self
            .blocks
            .get_mut(id.index())
            .ok_or_else(|| EngineError::Internal(format!("release of unknown block {id}")))?;
        if block.ref_count == 0 {
            return Err(EngineError::Internal(format!(
                "double release of {id}: ref_count is already 0"
            )));
        }
        block.ref_count -= 1;
        if block.ref_count == 0 {
            self.free_list.push_back(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Records how many tokens a block holds, and its hash once full.
    pub fn set_contents(
        &mut self,
        id: BlockId,
        num_tokens: usize,
        hash: Option<BlockHash>,
    ) -> Result<(), EngineError> {
        if num_tokens > self.block_size {
            return Err(EngineError::Internal(format!(
                "block {id} cannot hold {num_tokens} tokens, capacity is {}",
                self.block_size
            )));
        }
        let block = self
            .blocks
            .get_mut(id.index())
            .ok_or_else(|| EngineError::Internal(format!("set_contents on unknown block {id}")))?;
        block.num_tokens = num_tokens;
        block.hash = hash;
        Ok(())
    }

    /// Forgets the contents of the oldest free block that still holds any.
    ///
    /// Returns the block and the hash it carried, so the prefix cache can drop
    /// the corresponding index entry. `None` when no free block holds cached
    /// contents — that is, when eviction cannot make any further progress.
    ///
    /// Free blocks with no contents are skipped rather than treated as a stop
    /// condition: they are already immediately reusable, and the free list is
    /// ordered by release time, not by whether a block happens to be cached.
    pub fn evict_oldest_cached(&mut self) -> Option<(BlockId, BlockHash)> {
        let id = self
            .free_list
            .iter()
            .copied()
            .find(|id| self.blocks[id.index()].hash.is_some())?;

        let block = &mut self.blocks[id.index()];
        let hash = block.hash.take()?;
        block.num_tokens = 0;
        Some((id, hash))
    }

    /// Number of free blocks still holding cached contents, and therefore
    /// still evictable.
    pub fn num_evictable(&self) -> usize {
        self.free_list
            .iter()
            .filter(|id| self.blocks[id.index()].hash.is_some())
            .count()
    }
}

/// Removes `id` from the free list.
///
/// Linear in the free-list length. Called only on a prefix-cache hit, where the
/// alternative — an intrusive doubly-linked list threaded through the block
/// array — would need either `unsafe` or index bookkeeping that has to stay
/// consistent with the FIFO order. Measured against that complexity, the scan
/// is the better trade at current pool sizes; if profiling ever shows it, this
/// is the single place that changes.
fn remove_from_free_list(free_list: &mut VecDeque<BlockId>, id: BlockId) {
    if let Some(pos) = free_list.iter().position(|&b| b == id) {
        free_list.remove(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_pool_is_entirely_free() {
        let pool = BlockPool::new(8, 16);
        assert_eq!(pool.total_blocks(), 8);
        assert_eq!(pool.num_free(), 8);
        assert_eq!(pool.num_used(), 0);
        assert_eq!(pool.utilization(), 0.0);
    }

    #[test]
    fn allocation_takes_a_reference_and_consumes_a_free_block() {
        let mut pool = BlockPool::new(2, 16);
        let a = pool.allocate().unwrap().0;
        assert_eq!(pool.get(a).unwrap().ref_count(), 1);
        assert_eq!(pool.num_free(), 1);
        assert_eq!(pool.num_used(), 1);
    }

    #[test]
    fn exhausting_the_pool_reports_cache_exhausted() {
        let mut pool = BlockPool::new(1, 16);
        pool.allocate().unwrap();
        let err = pool.allocate().unwrap_err();
        assert!(matches!(err, EngineError::CacheExhausted { .. }));
        assert!(err.is_retryable());
    }

    #[test]
    fn a_block_is_reclaimed_only_when_the_last_reference_drops() {
        let mut pool = BlockPool::new(4, 16);
        let id = pool.allocate().unwrap().0;
        pool.add_ref(id).unwrap();
        assert_eq!(pool.get(id).unwrap().ref_count(), 2);

        assert!(!pool.release(id).unwrap(), "still referenced");
        assert_eq!(pool.num_free(), 3);
        assert!(pool.release(id).unwrap(), "last reference dropped");
        assert_eq!(pool.num_free(), 4);
    }

    #[test]
    fn double_release_is_an_internal_error_not_a_silent_underflow() {
        let mut pool = BlockPool::new(2, 16);
        let id = pool.allocate().unwrap().0;
        pool.release(id).unwrap();
        let err = pool.release(id).unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
    }

    #[test]
    fn a_freed_block_keeps_its_contents_until_reallocated() {
        let mut pool = BlockPool::new(2, 16);
        let id = pool.allocate().unwrap().0;
        pool.set_contents(id, 16, Some(BlockHash(0xabc))).unwrap();
        pool.release(id).unwrap();

        // Contents survive so the prefix cache can still claim the block.
        assert_eq!(pool.get(id).unwrap().hash(), Some(BlockHash(0xabc)));
        assert_eq!(pool.get(id).unwrap().num_tokens(), 16);
    }

    #[test]
    fn reviving_a_freed_block_removes_it_from_the_free_list() {
        let mut pool = BlockPool::new(4, 16);
        let id = pool.allocate().unwrap().0;
        pool.set_contents(id, 16, Some(BlockHash(1))).unwrap();
        pool.release(id).unwrap();
        assert_eq!(pool.num_free(), 4);

        pool.add_ref(id).unwrap();
        assert_eq!(pool.num_free(), 3, "revived block must leave the free list");
        assert_eq!(pool.get(id).unwrap().ref_count(), 1);
        assert_eq!(
            pool.get(id).unwrap().hash(),
            Some(BlockHash(1)),
            "revival must preserve cached contents"
        );
    }

    #[test]
    fn allocation_reports_the_hash_it_recycled() {
        let mut pool = BlockPool::new(1, 16);
        let (id, evicted) = pool.allocate().unwrap();
        assert_eq!(evicted, None, "a never-used block evicts nothing");

        pool.set_contents(id, 16, Some(BlockHash(0x1234))).unwrap();
        pool.release(id).unwrap();

        // Recycling the block must report the hash whose index entry is now
        // stale, so the caller can drop it.
        let (again, evicted) = pool.allocate().unwrap();
        assert_eq!(again, id);
        assert_eq!(evicted, Some(BlockHash(0x1234)));
    }

    #[test]
    fn reallocation_clears_stale_contents() {
        let mut pool = BlockPool::new(1, 16);
        let id = pool.allocate().unwrap().0;
        pool.set_contents(id, 16, Some(BlockHash(7))).unwrap();
        pool.release(id).unwrap();

        let again = pool.allocate().unwrap().0;
        assert_eq!(again, id, "single-block pool must hand back the same block");
        assert_eq!(pool.get(id).unwrap().hash(), None);
        assert_eq!(pool.get(id).unwrap().num_tokens(), 0);
    }

    #[test]
    fn free_blocks_are_recycled_oldest_first() {
        let mut pool = BlockPool::new(3, 16);
        let a = pool.allocate().unwrap().0;
        let b = pool.allocate().unwrap().0;
        let c = pool.allocate().unwrap().0;

        // Release out of allocation order; recycling must follow release order.
        pool.release(c).unwrap();
        pool.release(a).unwrap();
        pool.release(b).unwrap();

        assert_eq!(pool.allocate().unwrap().0, c);
        assert_eq!(pool.allocate().unwrap().0, a);
        assert_eq!(pool.allocate().unwrap().0, b);
    }

    #[test]
    fn only_full_hashed_blocks_are_shareable() {
        let mut pool = BlockPool::new(2, 16);
        let id = pool.allocate().unwrap().0;

        pool.set_contents(id, 8, None).unwrap();
        assert!(!pool.get(id).unwrap().is_shareable(16), "partial block");

        pool.set_contents(id, 16, None).unwrap();
        assert!(!pool.get(id).unwrap().is_shareable(16), "full but unhashed");

        pool.set_contents(id, 16, Some(BlockHash(3))).unwrap();
        assert!(pool.get(id).unwrap().is_shareable(16));
    }

    #[test]
    fn overfilling_a_block_is_rejected() {
        let mut pool = BlockPool::new(1, 16);
        let id = pool.allocate().unwrap().0;
        assert!(pool.set_contents(id, 17, None).is_err());
    }

    #[test]
    fn eviction_forgets_the_contents_of_a_cached_free_block() {
        let mut pool = BlockPool::new(2, 16);
        let a = pool.allocate().unwrap().0;
        pool.set_contents(a, 16, Some(BlockHash(9))).unwrap();
        pool.release(a).unwrap();
        assert_eq!(pool.num_evictable(), 1);

        let (evicted, hash) = pool.evict_oldest_cached().unwrap();
        assert_eq!(evicted, a);
        assert_eq!(hash, BlockHash(9));
        assert_eq!(pool.get(a).unwrap().hash(), None);
        assert_eq!(pool.num_evictable(), 0);
    }

    #[test]
    fn eviction_skips_free_blocks_that_hold_nothing() {
        // Block 1 is never used and sits at the front of the free list; the
        // cached block behind it must still be reachable for eviction.
        let mut pool = BlockPool::new(2, 16);
        let a = pool.allocate().unwrap().0;
        assert_eq!(a, BlockId(0));
        pool.set_contents(a, 16, Some(BlockHash(9))).unwrap();
        pool.release(a).unwrap();

        let (evicted, hash) = pool.evict_oldest_cached().unwrap();
        assert_eq!(evicted, BlockId(0));
        assert_eq!(hash, BlockHash(9));
    }

    #[test]
    fn eviction_reports_exhaustion_when_nothing_is_cached() {
        let mut pool = BlockPool::new(2, 16);
        assert!(pool.evict_oldest_cached().is_none());
        assert_eq!(pool.num_evictable(), 0);
    }

    #[test]
    fn a_referenced_block_is_never_evicted() {
        let mut pool = BlockPool::new(2, 16);
        let a = pool.allocate().unwrap().0;
        pool.set_contents(a, 16, Some(BlockHash(9))).unwrap();
        // Still referenced, so not on the free list and not evictable.
        assert_eq!(pool.num_evictable(), 0);
        assert!(pool.evict_oldest_cached().is_none());
        assert_eq!(pool.get(a).unwrap().hash(), Some(BlockHash(9)));
    }

    #[test]
    fn utilization_tracks_referenced_blocks() {
        let mut pool = BlockPool::new(4, 16);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        assert_eq!(pool.utilization(), 0.5);
    }
}
