//! Paged attention: the operation that reads and writes the block-structured
//! KV cache.
//!
//! This is where the paged cache design meets the actual arithmetic. A
//! conventional attention kernel reads keys and values from one contiguous
//! buffer per sequence. Here they are scattered across physical blocks that a
//! block table maps to, so every access goes through one level of indirection.
//!
//! # Layout
//!
//! The KV arena is a single flat allocation, indexed as
//!
//! ```text
//! [layer][block][k_or_v][slot_in_block][head][dim]
//! ```
//!
//! Layer is outermost because a forward pass touches all blocks of one layer
//! before moving to the next, so this ordering keeps a layer's working set
//! contiguous. Within a block, `slot` precedes `head` so that the `head_dim`
//! values of one head at one position are adjacent — that is the innermost
//! loop of the dot product, and it is the access that most needs to be
//! contiguous.

use orion_core::EngineError;

use crate::tensor::softmax_inplace;

/// Flat storage for all keys and values across all layers and blocks.
///
/// Sized once at startup from the block count; it never grows. That is the
/// point of the paged design — memory is committed up front and then managed,
/// rather than allocated per request.
#[derive(Debug)]
pub struct KvStore {
    data: Vec<f32>,
    num_layers: usize,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KvStore {
    /// Allocates the arena.
    pub fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        // 2 for keys and values.
        let total = num_layers * num_blocks * 2 * block_size * num_kv_heads * head_dim;
        Self {
            data: vec![0.0; total],
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        }
    }

    /// Total elements held. Multiply by 4 for bytes at `f32`.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Elements spanned by one block's keys (or values) for one layer.
    fn block_stride(&self) -> usize {
        self.block_size * self.num_kv_heads * self.head_dim
    }

    /// Offset of a (layer, block, key-or-value) region.
    fn region_offset(&self, layer: usize, block: u32, is_value: bool) -> usize {
        let per_block = 2 * self.block_stride();
        let per_layer = self.num_blocks * per_block;
        layer * per_layer + block as usize * per_block + usize::from(is_value) * self.block_stride()
    }

    /// Offset of one head's vector at one slot within a block.
    fn slot_offset(
        &self,
        layer: usize,
        block: u32,
        is_value: bool,
        slot: usize,
        head: usize,
    ) -> usize {
        self.region_offset(layer, block, is_value)
            + (slot * self.num_kv_heads + head) * self.head_dim
    }

    /// Writes one key or value vector into the cache.
    pub fn write(
        &mut self,
        layer: usize,
        block: u32,
        slot: usize,
        head: usize,
        is_value: bool,
        vector: &[f32],
    ) -> Result<(), EngineError> {
        self.check_bounds(layer, block, slot, head, vector.len())?;
        let off = self.slot_offset(layer, block, is_value, slot, head);
        self.data[off..off + self.head_dim].copy_from_slice(vector);
        Ok(())
    }

    /// Reads one key or value vector from the cache.
    pub fn read(
        &self,
        layer: usize,
        block: u32,
        slot: usize,
        head: usize,
        is_value: bool,
    ) -> Result<&[f32], EngineError> {
        self.check_bounds(layer, block, slot, head, self.head_dim)?;
        let off = self.slot_offset(layer, block, is_value, slot, head);
        Ok(&self.data[off..off + self.head_dim])
    }

    fn check_bounds(
        &self,
        layer: usize,
        block: u32,
        slot: usize,
        head: usize,
        vec_len: usize,
    ) -> Result<(), EngineError> {
        if layer >= self.num_layers {
            return Err(EngineError::Internal(format!(
                "layer {layer} out of range (have {})",
                self.num_layers
            )));
        }
        if block as usize >= self.num_blocks {
            return Err(EngineError::Internal(format!(
                "block {block} out of range (have {})",
                self.num_blocks
            )));
        }
        if slot >= self.block_size {
            return Err(EngineError::Internal(format!(
                "slot {slot} out of range (block holds {})",
                self.block_size
            )));
        }
        if head >= self.num_kv_heads {
            return Err(EngineError::Internal(format!(
                "kv head {head} out of range (have {})",
                self.num_kv_heads
            )));
        }
        if vec_len != self.head_dim {
            return Err(EngineError::Internal(format!(
                "vector length {vec_len} does not match head_dim {}",
                self.head_dim
            )));
        }
        Ok(())
    }
}

/// Resolves a logical token position to its physical (block, slot).
///
/// The whole indirection of the paged design lives in this one function.
#[inline]
pub fn locate(block_table: &[u32], position: usize, block_size: usize) -> Option<(u32, usize)> {
    let logical_block = position / block_size;
    let slot = position % block_size;
    block_table.get(logical_block).map(|&b| (b, slot))
}

/// Arguments for one sequence's attention over the paged cache.
#[derive(Debug, Clone, Copy)]
pub struct AttentionArgs<'a> {
    /// Physical blocks backing this sequence, in logical order.
    pub block_table: &'a [u32],
    /// Total context length including the tokens being added this step.
    pub context_len: usize,
    /// Absolute position of the query token.
    pub query_position: usize,
    pub layer: usize,
    pub num_query_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
}

/// Computes attention for one query token against the cached keys and values.
///
/// `query` is `[num_query_heads * head_dim]`; output is the same shape.
///
/// # Causal masking
///
/// Only positions `0..=query_position` are attended to. Masking is expressed
/// by *not iterating* over future positions rather than by adding `-inf` to
/// their scores — there is nothing to mask when the loop never visits them,
/// and it avoids the `exp(-inf - -inf) = NaN` hazard entirely.
///
/// # Grouped-query attention
///
/// Query head `h` reads KV head `h / group_size`. Several query heads share one
/// KV head, which is what makes the cache smaller by exactly that factor.
pub fn paged_attention(
    store: &KvStore,
    query: &[f32],
    args: AttentionArgs<'_>,
    out: &mut [f32],
) -> Result<(), EngineError> {
    let AttentionArgs {
        block_table,
        query_position,
        layer,
        num_query_heads,
        num_kv_heads,
        head_dim,
        block_size,
        ..
    } = args;

    if query.len() != num_query_heads * head_dim || out.len() != query.len() {
        return Err(EngineError::Internal(format!(
            "attention expected {} query elements, got {} (out {})",
            num_query_heads * head_dim,
            query.len(),
            out.len()
        )));
    }
    if num_kv_heads == 0 || num_query_heads % num_kv_heads != 0 {
        return Err(EngineError::Internal(format!(
            "{num_query_heads} query heads cannot be grouped over {num_kv_heads} kv heads"
        )));
    }

    let group_size = num_query_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Causal: attend to everything up to and including the query's position.
    let attend_len = query_position + 1;

    let mut scores = vec![0.0f32; attend_len];

    for qh in 0..num_query_heads {
        let kv_head = qh / group_size;
        let q = &query[qh * head_dim..(qh + 1) * head_dim];

        // Scores against every key up to the causal limit.
        for (pos, score) in scores.iter_mut().enumerate() {
            let (block, slot) = locate(block_table, pos, block_size).ok_or_else(|| {
                EngineError::Internal(format!(
                    "position {pos} is outside the block table of {} blocks",
                    block_table.len()
                ))
            })?;
            let k = store.read(layer, block, slot, kv_head, false)?;
            let mut dot = 0.0f32;
            for (a, b) in q.iter().zip(k.iter()) {
                dot += a * b;
            }
            *score = dot * scale;
        }

        softmax_inplace(&mut scores);

        // Weighted sum of values.
        let o = &mut out[qh * head_dim..(qh + 1) * head_dim];
        o.fill(0.0);
        for (pos, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let (block, slot) = locate(block_table, pos, block_size)
                .ok_or_else(|| EngineError::Internal(format!("position {pos} unmapped")))?;
            let v = store.read(layer, block, slot, kv_head, true)?;
            for (acc, &val) in o.iter_mut().zip(v.iter()) {
                *acc += w * val;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> KvStore {
        // 2 layers, 4 blocks, block size 2, 2 kv heads, head dim 4.
        KvStore::new(2, 4, 2, 2, 4)
    }

    #[test]
    fn the_arena_is_sized_from_its_dimensions() {
        let s = store();
        assert_eq!(s.len(), 2 * 4 * 2 * 2 * 2 * 4);
        assert_eq!(s.block_size(), 2);
        assert_eq!(s.num_blocks(), 4);
    }

    #[test]
    fn a_written_vector_reads_back_unchanged() {
        let mut s = store();
        s.write(1, 2, 1, 0, false, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(s.read(1, 2, 1, 0, false).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn keys_and_values_occupy_distinct_storage() {
        let mut s = store();
        s.write(0, 0, 0, 0, false, &[1.0, 1.0, 1.0, 1.0]).unwrap();
        s.write(0, 0, 0, 0, true, &[2.0, 2.0, 2.0, 2.0]).unwrap();
        assert_eq!(s.read(0, 0, 0, 0, false).unwrap()[0], 1.0);
        assert_eq!(s.read(0, 0, 0, 0, true).unwrap()[0], 2.0);
    }

    #[test]
    fn every_index_dimension_is_independent() {
        // Writing to one (layer, block, slot, head) must not disturb another.
        let mut s = store();
        s.write(0, 0, 0, 0, false, &[1.0; 4]).unwrap();
        s.write(1, 0, 0, 0, false, &[2.0; 4]).unwrap();
        s.write(0, 1, 0, 0, false, &[3.0; 4]).unwrap();
        s.write(0, 0, 1, 0, false, &[4.0; 4]).unwrap();
        s.write(0, 0, 0, 1, false, &[5.0; 4]).unwrap();

        assert_eq!(s.read(0, 0, 0, 0, false).unwrap()[0], 1.0);
        assert_eq!(s.read(1, 0, 0, 0, false).unwrap()[0], 2.0);
        assert_eq!(s.read(0, 1, 0, 0, false).unwrap()[0], 3.0);
        assert_eq!(s.read(0, 0, 1, 0, false).unwrap()[0], 4.0);
        assert_eq!(s.read(0, 0, 0, 1, false).unwrap()[0], 5.0);
    }

    #[test]
    fn out_of_range_indices_are_errors_not_panics() {
        let mut s = store();
        assert!(s.write(9, 0, 0, 0, false, &[0.0; 4]).is_err(), "layer");
        assert!(s.write(0, 9, 0, 0, false, &[0.0; 4]).is_err(), "block");
        assert!(s.write(0, 0, 9, 0, false, &[0.0; 4]).is_err(), "slot");
        assert!(s.write(0, 0, 0, 9, false, &[0.0; 4]).is_err(), "head");
        assert!(s.write(0, 0, 0, 0, false, &[0.0; 3]).is_err(), "length");
    }

    #[test]
    fn locate_maps_logical_positions_through_the_block_table() {
        let table = [7u32, 3, 9];
        // block_size 4: positions 0-3 -> block 7, 4-7 -> block 3, 8-11 -> 9.
        assert_eq!(locate(&table, 0, 4), Some((7, 0)));
        assert_eq!(locate(&table, 3, 4), Some((7, 3)));
        assert_eq!(locate(&table, 4, 4), Some((3, 0)));
        assert_eq!(locate(&table, 11, 4), Some((9, 3)));
        assert_eq!(locate(&table, 12, 4), None, "beyond the table");
    }

    #[test]
    fn locate_handles_non_contiguous_physical_blocks() {
        // The whole point of paging: physical order need not match logical.
        let table = [100u32, 2, 57];
        assert_eq!(locate(&table, 5, 4), Some((2, 1)));
    }

    /// Attention over a single position must return exactly that position's
    /// value, since softmax over one score is 1.0.
    #[test]
    fn attention_over_one_position_returns_that_value() {
        let mut s = KvStore::new(1, 2, 4, 1, 4);
        s.write(0, 0, 0, 0, false, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        s.write(0, 0, 0, 0, true, &[9.0, 8.0, 7.0, 6.0]).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0; 4];
        paged_attention(
            &s,
            &query,
            AttentionArgs {
                block_table: &[0],
                context_len: 1,
                query_position: 0,
                layer: 0,
                num_query_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                block_size: 4,
            },
            &mut out,
        )
        .unwrap();

        assert_eq!(out, vec![9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn attention_output_is_a_convex_combination_of_values() {
        // Softmax weights sum to 1, so the output must lie within the range of
        // the values attended to.
        let mut s = KvStore::new(1, 2, 4, 1, 2);
        for pos in 0..4 {
            s.write(0, 0, pos, 0, false, &[pos as f32, 0.0]).unwrap();
            s.write(0, 0, pos, 0, true, &[(pos as f32) * 10.0, 1.0])
                .unwrap();
        }

        let mut out = vec![0.0; 2];
        paged_attention(
            &s,
            &[1.0, 0.0],
            AttentionArgs {
                block_table: &[0],
                context_len: 4,
                query_position: 3,
                layer: 0,
                num_query_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                block_size: 4,
            },
            &mut out,
        )
        .unwrap();

        assert!(
            out[0] >= 0.0 && out[0] <= 30.0,
            "out of value range: {out:?}"
        );
        // Every value's second element is 1.0, so a convex combination is 1.0.
        assert!((out[1] - 1.0).abs() < 1e-5, "weights should sum to 1");
    }

    #[test]
    fn causal_masking_ignores_future_positions() {
        let mut s = KvStore::new(1, 4, 4, 1, 2);
        // Position 0 has a benign value; positions 1-3 have a huge one.
        s.write(0, 0, 0, 0, false, &[1.0, 0.0]).unwrap();
        s.write(0, 0, 0, 0, true, &[5.0, 5.0]).unwrap();
        for pos in 1..4 {
            s.write(0, 0, pos, 0, false, &[1.0, 0.0]).unwrap();
            s.write(0, 0, pos, 0, true, &[1000.0, 1000.0]).unwrap();
        }

        // Querying at position 0 must not see positions 1-3.
        let mut out = vec![0.0; 2];
        paged_attention(
            &s,
            &[1.0, 0.0],
            AttentionArgs {
                block_table: &[0],
                context_len: 4,
                query_position: 0,
                layer: 0,
                num_query_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                block_size: 4,
            },
            &mut out,
        )
        .unwrap();

        assert_eq!(
            out,
            vec![5.0, 5.0],
            "future positions leaked into the output"
        );
    }

    #[test]
    fn attention_spans_multiple_blocks() {
        // block_size 2, so positions 0-1 live in block 0 and 2-3 in block 1.
        let mut s = KvStore::new(1, 4, 2, 1, 2);
        let blocks = [0u32, 1];
        for pos in 0..4 {
            let (b, slot) = locate(&blocks, pos, 2).unwrap();
            s.write(0, b, slot, 0, false, &[1.0, 0.0]).unwrap();
            s.write(0, b, slot, 0, true, &[pos as f32, 1.0]).unwrap();
        }

        let mut out = vec![0.0; 2];
        paged_attention(
            &s,
            &[1.0, 0.0],
            AttentionArgs {
                block_table: &blocks,
                context_len: 4,
                query_position: 3,
                layer: 0,
                num_query_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                block_size: 2,
            },
            &mut out,
        )
        .unwrap();

        // All keys identical, so weights are uniform: mean of 0,1,2,3 = 1.5.
        assert!((out[0] - 1.5).abs() < 1e-5, "got {out:?}");
    }

    #[test]
    fn a_scattered_block_table_gives_the_same_answer_as_a_contiguous_one() {
        // Paging must be transparent to the arithmetic.
        let run = |blocks: &[u32]| {
            let mut s = KvStore::new(1, 8, 2, 1, 2);
            for pos in 0..4 {
                let (b, slot) = locate(blocks, pos, 2).unwrap();
                s.write(0, b, slot, 0, false, &[pos as f32 * 0.5, 1.0])
                    .unwrap();
                s.write(0, b, slot, 0, true, &[pos as f32, 2.0]).unwrap();
            }
            let mut out = vec![0.0; 2];
            paged_attention(
                &s,
                &[0.7, 0.3],
                AttentionArgs {
                    block_table: blocks,
                    context_len: 4,
                    query_position: 3,
                    layer: 0,
                    num_query_heads: 1,
                    num_kv_heads: 1,
                    head_dim: 2,
                    block_size: 2,
                },
                &mut out,
            )
            .unwrap();
            out
        };

        let contiguous = run(&[0, 1]);
        let scattered = run(&[6, 2]);
        for (a, b) in contiguous.iter().zip(scattered.iter()) {
            assert!((a - b).abs() < 1e-6, "{contiguous:?} vs {scattered:?}");
        }
    }

    #[test]
    fn grouped_query_heads_share_kv_heads() {
        // 4 query heads over 2 KV heads: q0,q1 -> kv0 and q2,q3 -> kv1.
        let mut s = KvStore::new(1, 1, 2, 2, 2);
        s.write(0, 0, 0, 0, false, &[1.0, 0.0]).unwrap();
        s.write(0, 0, 0, 0, true, &[10.0, 10.0]).unwrap();
        s.write(0, 0, 0, 1, false, &[1.0, 0.0]).unwrap();
        s.write(0, 0, 0, 1, true, &[20.0, 20.0]).unwrap();

        let query = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0; 8];
        paged_attention(
            &s,
            &query,
            AttentionArgs {
                block_table: &[0],
                context_len: 1,
                query_position: 0,
                layer: 0,
                num_query_heads: 4,
                num_kv_heads: 2,
                head_dim: 2,
                block_size: 2,
            },
            &mut out,
        )
        .unwrap();

        assert_eq!(&out[0..2], &[10.0, 10.0], "q0 reads kv0");
        assert_eq!(&out[2..4], &[10.0, 10.0], "q1 also reads kv0");
        assert_eq!(&out[4..6], &[20.0, 20.0], "q2 reads kv1");
        assert_eq!(&out[6..8], &[20.0, 20.0], "q3 also reads kv1");
    }

    #[test]
    fn attention_favours_the_most_similar_key() {
        let mut s = KvStore::new(1, 1, 4, 1, 2);
        // Position 1's key aligns with the query; the others do not.
        s.write(0, 0, 0, 0, false, &[0.0, 1.0]).unwrap();
        s.write(0, 0, 0, 0, true, &[1.0, 0.0]).unwrap();
        s.write(0, 0, 1, 0, false, &[10.0, 0.0]).unwrap();
        s.write(0, 0, 1, 0, true, &[0.0, 1.0]).unwrap();

        let mut out = vec![0.0; 2];
        paged_attention(
            &s,
            &[10.0, 0.0],
            AttentionArgs {
                block_table: &[0],
                context_len: 2,
                query_position: 1,
                layer: 0,
                num_query_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                block_size: 4,
            },
            &mut out,
        )
        .unwrap();

        assert!(out[1] > 0.9, "should strongly prefer position 1: {out:?}");
        assert!(out[0] < 0.1);
    }

    #[test]
    fn mismatched_head_grouping_is_rejected() {
        let s = KvStore::new(1, 1, 2, 2, 2);
        let mut out = vec![0.0; 6];
        let err = paged_attention(
            &s,
            &[0.0; 6],
            AttentionArgs {
                block_table: &[0],
                context_len: 1,
                query_position: 0,
                layer: 0,
                num_query_heads: 3,
                num_kv_heads: 2,
                head_dim: 2,
                block_size: 2,
            },
            &mut out,
        );
        assert!(err.is_err(), "3 query heads do not divide over 2 kv heads");
    }

    #[test]
    fn a_position_outside_the_block_table_is_an_error() {
        let s = KvStore::new(1, 1, 2, 1, 2);
        let mut out = vec![0.0; 2];
        // Position 5 needs a third block; the table has one.
        let err = paged_attention(
            &s,
            &[1.0, 0.0],
            AttentionArgs {
                block_table: &[0],
                context_len: 6,
                query_position: 5,
                layer: 0,
                num_query_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                block_size: 2,
            },
            &mut out,
        );
        assert!(matches!(err, Err(EngineError::Internal(_))));
    }
}
