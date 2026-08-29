//! Tensor-parallel execution and GPU collective communication.
//!
//! # Status: not implemented, not verified
//!
//! **No multi-GPU code has been executed.** The development machine has no
//! NVIDIA GPU at all, let alone several, and no NCCL installation. What follows
//! is a design with a tested *partitioning calculus* — the arithmetic that
//! decides how weights are split, which is verifiable without hardware — and
//! stubs for everything that needs a device.
//!
//! Nothing here claims a speedup. Scaling efficiency is a measurement, and no
//! measurement has been taken.
//!
//! # Tensor parallelism in one paragraph
//!
//! Each linear layer is split across GPUs, every GPU holds a slice of the
//! weights, and a collective operation stitches the partial results back
//! together. Unlike pipeline parallelism it adds no bubble and keeps every GPU
//! busy on every token — at the cost of a collective on the critical path of
//! every layer, which is why interconnect bandwidth decides whether it is worth
//! doing at all.
//!
//! # Why the split points are what they are
//!
//! ```text
//!        x  (replicated on every rank)
//!        │
//!   ┌────┴────┐   column-parallel: split the OUTPUT dimension
//!   ▼         ▼   no communication needed, each rank has full x
//!  W_a       W_b
//!   │         │
//!   ▼         ▼   partial outputs, each [tokens, out/N]
//!  y_a       y_b
//!   │         │
//!   └────┬────┘   row-parallel: split the INPUT dimension
//!        ▼        each rank produces a partial sum
//!    AllReduce    <-- the one collective per layer
//!        │
//!        ▼
//!        y
//! ```
//!
//! Pairing a column-parallel layer with a row-parallel one means the
//! intermediate never needs gathering: rank `i`'s slice of the first output is
//! exactly the input its slice of the second weight needs. **One** AllReduce
//! per attention block and one per MLP block, rather than two of each.
//!
//! For attention this means splitting by *head*: each rank owns whole heads, so
//! the softmax — which must see a complete head — needs no communication.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use orion_core::{EngineError, ModelMetadata};

/// How a weight matrix is split across ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Split the output dimension. Input is replicated; no collective needed
    /// before the next operation.
    ColumnParallel,
    /// Split the input dimension. Each rank produces a partial sum, requiring
    /// an AllReduce.
    RowParallel,
    /// Not split; every rank holds the full copy.
    ///
    /// Used for small tensors — norms, biases — where the memory saved would be
    /// dwarfed by the communication cost of splitting.
    Replicated,
}

/// A rank's position in the tensor-parallel group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank {
    pub index: usize,
    pub world_size: usize,
}

impl Rank {
    pub fn new(index: usize, world_size: usize) -> Result<Self, EngineError> {
        if world_size == 0 {
            return Err(EngineError::InvalidRequest(
                "world_size must be at least 1".into(),
            ));
        }
        if index >= world_size {
            return Err(EngineError::InvalidRequest(format!(
                "rank {index} is out of range for world size {world_size}"
            )));
        }
        Ok(Self { index, world_size })
    }

    /// A single-GPU (or CPU) run, which is the degenerate case of the same
    /// code path rather than a separate one.
    pub fn single() -> Self {
        Self {
            index: 0,
            world_size: 1,
        }
    }

    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }

    /// Whether this rank does work that must not be duplicated — logging,
    /// metrics, writing output.
    pub fn is_leader(&self) -> bool {
        self.index == 0
    }
}

/// The slice of a dimension one rank owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shard {
    pub start: usize,
    pub len: usize,
}

impl Shard {
    pub fn end(&self) -> usize {
        self.start + self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Splits `total` elements across `world_size` ranks.
///
/// Requires exact divisibility rather than distributing a remainder. Uneven
/// shards would make every rank's kernel launch a different shape and leave the
/// largest rank as a straggler that every collective waits on — the slowest
/// rank sets the pace for all of them. Failing loudly at configuration time is
/// far better than silently accepting a layout that will underperform on every
/// token.
pub fn shard_dimension(total: usize, rank: Rank) -> Result<Shard, EngineError> {
    if !total.is_multiple_of(rank.world_size) {
        return Err(EngineError::InvalidRequest(format!(
            "cannot split {total} evenly across {} ranks; \
             tensor parallelism requires the dimension to be divisible by the world size",
            rank.world_size
        )));
    }
    let len = total / rank.world_size;
    Ok(Shard {
        start: rank.index * len,
        len,
    })
}

/// How a model's weights are laid out across a tensor-parallel group.
#[derive(Debug, Clone)]
pub struct ParallelLayout {
    pub rank: Rank,
    /// Attention heads this rank owns.
    pub query_heads: Shard,
    /// KV heads this rank owns.
    pub kv_heads: Shard,
    /// Slice of the MLP intermediate dimension this rank owns.
    pub ffn: Shard,
}

impl ParallelLayout {
    /// Computes the layout for one rank, validating that the model can be split.
    ///
    /// The KV-head constraint is the one that bites in practice: grouped-query
    /// attention models often have few KV heads (Llama-3-8B has 8), so they
    /// cannot be split across more ranks than they have KV heads without
    /// replicating them. This reports that as a configuration error rather than
    /// silently choosing a layout the operator did not ask for.
    pub fn compute(meta: &ModelMetadata, rank: Rank, ffn_dim: usize) -> Result<Self, EngineError> {
        if !rank.is_distributed() {
            return Ok(Self {
                rank,
                query_heads: Shard {
                    start: 0,
                    len: meta.num_attention_heads,
                },
                kv_heads: Shard {
                    start: 0,
                    len: meta.num_kv_heads,
                },
                ffn: Shard {
                    start: 0,
                    len: ffn_dim,
                },
            });
        }

        if meta.num_kv_heads < rank.world_size {
            return Err(EngineError::InvalidRequest(format!(
                "model has {} KV heads but world size is {}; \
                 tensor parallelism beyond the KV head count would require replicating them, \
                 which is not implemented",
                meta.num_kv_heads, rank.world_size
            )));
        }

        Ok(Self {
            rank,
            query_heads: shard_dimension(meta.num_attention_heads, rank)?,
            kv_heads: shard_dimension(meta.num_kv_heads, rank)?,
            ffn: shard_dimension(ffn_dim, rank)?,
        })
    }

    /// Hidden-dimension width this rank's attention output occupies.
    pub fn local_attention_dim(&self, head_dim: usize) -> usize {
        self.query_heads.len * head_dim
    }

    /// KV cache this rank holds per token, in bytes.
    ///
    /// The saving that makes tensor parallelism attractive for long context:
    /// each rank stores only its own KV heads, so cache memory per GPU falls
    /// linearly with world size.
    pub fn local_kv_bytes_per_token(&self, meta: &ModelMetadata) -> usize {
        let elems = 2 * meta.num_layers * self.kv_heads.len * meta.head_dim;
        meta.dtype.size_in_bytes(elems)
    }
}

/// Collective operations a tensor-parallel forward pass needs.
///
/// Defined as a trait so the partitioning logic above can be tested against a
/// single-rank no-op implementation, with no NCCL and no GPU. That is the only
/// implementation that currently exists.
pub trait Collective: std::fmt::Debug + Send + Sync {
    fn rank(&self) -> Rank;

    /// Sums a buffer across all ranks, leaving the total on every rank.
    ///
    /// The one collective on the critical path of every layer. Its latency,
    /// not its bandwidth, usually dominates: the buffers are small
    /// (`tokens × hidden`) and the operation happens twice per layer, so a
    /// 32-layer model performs 64 of them per token.
    fn all_reduce_sum(&self, buffer: &mut [f32]) -> Result<(), EngineError>;

    /// Concatenates each rank's slice into a full buffer on every rank.
    fn all_gather(&self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError>;

    /// Sums across ranks and leaves each rank with one slice of the result.
    ///
    /// Half the traffic of an AllReduce when the consumer only needs its own
    /// slice, which is the case when a row-parallel layer feeds a
    /// column-parallel one.
    fn reduce_scatter_sum(&self, input: &[f32], out: &mut [f32]) -> Result<(), EngineError>;

    /// Blocks until every rank reaches this point.
    fn barrier(&self) -> Result<(), EngineError>;
}

/// The single-rank implementation: every collective is a no-op or a copy.
///
/// Not a placeholder — it is the correct implementation for `world_size == 1`,
/// and it lets the same code path serve single-GPU and multi-GPU runs. Having
/// one path rather than two means the distributed path is exercised by every
/// single-GPU test.
#[derive(Debug, Clone)]
pub struct SingleRank;

impl Collective for SingleRank {
    fn rank(&self) -> Rank {
        Rank::single()
    }

    /// Summing across one rank leaves the buffer unchanged.
    fn all_reduce_sum(&self, _buffer: &mut [f32]) -> Result<(), EngineError> {
        Ok(())
    }

    fn all_gather(&self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        if local.len() != out.len() {
            return Err(EngineError::Internal(format!(
                "all_gather on a single rank expects equal sizes, got {} and {}",
                local.len(),
                out.len()
            )));
        }
        out.copy_from_slice(local);
        Ok(())
    }

    fn reduce_scatter_sum(&self, input: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        if input.len() != out.len() {
            return Err(EngineError::Internal(format!(
                "reduce_scatter on a single rank expects equal sizes, got {} and {}",
                input.len(),
                out.len()
            )));
        }
        out.copy_from_slice(input);
        Ok(())
    }

    fn barrier(&self) -> Result<(), EngineError> {
        Ok(())
    }
}

/// Estimates the communication volume of one forward pass.
///
/// An **estimate from the layer structure**, not a measurement. It is useful
/// for deciding whether tensor parallelism is worth attempting on a given
/// interconnect before writing any of it — if the estimate says the collectives
/// will dominate, they will.
#[derive(Debug, Clone, Copy)]
pub struct CommunicationEstimate {
    pub all_reduce_count: usize,
    pub bytes_per_token: usize,
}

impl CommunicationEstimate {
    /// Two AllReduces per layer: one after attention's row-parallel output
    /// projection, one after the MLP's row-parallel down projection.
    pub fn for_model(meta: &ModelMetadata, world_size: usize) -> Self {
        if world_size <= 1 {
            return Self {
                all_reduce_count: 0,
                bytes_per_token: 0,
            };
        }
        let per_layer = 2;
        let count = meta.num_layers * per_layer;
        // A ring AllReduce moves roughly 2(N-1)/N times the buffer per rank.
        let buffer = meta.dtype.size_in_bytes(meta.hidden_size);
        let factor = 2.0 * (world_size - 1) as f64 / world_size as f64;
        Self {
            all_reduce_count: count,
            bytes_per_token: (count as f64 * buffer as f64 * factor) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_core::DType;

    fn llama_8b() -> ModelMetadata {
        ModelMetadata {
            architecture: "llama".into(),
            name: "llama-3-8b".into(),
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 128256,
            max_position_embeddings: 8192,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            dtype: DType::F16,
            eos_token_ids: vec![128001],
            bos_token_id: Some(128000),
        }
    }

    #[test]
    fn ranks_are_validated_on_construction() {
        assert!(Rank::new(0, 1).is_ok());
        assert!(Rank::new(3, 4).is_ok());
        assert!(Rank::new(4, 4).is_err(), "rank must be below world size");
        assert!(Rank::new(0, 0).is_err(), "world size must be positive");
    }

    #[test]
    fn a_single_rank_is_not_distributed_and_leads() {
        let r = Rank::single();
        assert!(!r.is_distributed());
        assert!(r.is_leader());
        assert_eq!(r.world_size, 1);
    }

    #[test]
    fn only_rank_zero_leads() {
        assert!(Rank::new(0, 4).unwrap().is_leader());
        for i in 1..4 {
            assert!(!Rank::new(i, 4).unwrap().is_leader());
        }
    }

    #[test]
    fn shards_partition_a_dimension_exactly() {
        // Every element belongs to exactly one rank, with no gaps or overlaps.
        let total = 32;
        let world = 4;
        let shards: Vec<Shard> = (0..world)
            .map(|i| shard_dimension(total, Rank::new(i, world).unwrap()).unwrap())
            .collect();

        assert_eq!(shards[0], Shard { start: 0, len: 8 });
        assert_eq!(shards[3], Shard { start: 24, len: 8 });
        assert_eq!(shards.iter().map(|s| s.len).sum::<usize>(), total);

        for pair in shards.windows(2) {
            assert_eq!(pair[0].end(), pair[1].start, "shards must be contiguous");
        }
    }

    #[test]
    fn uneven_splits_are_refused_rather_than_rounded() {
        // An uneven split leaves the largest rank as a straggler that every
        // collective waits on. Better to fail at configuration time.
        let err = shard_dimension(30, Rank::new(0, 4).unwrap()).unwrap_err();
        assert!(matches!(err, EngineError::InvalidRequest(_)));
        assert!(err.to_string().contains("30"));
        assert!(err.to_string().contains('4'));
    }

    #[test]
    fn a_single_rank_owns_everything() {
        let m = llama_8b();
        let layout = ParallelLayout::compute(&m, Rank::single(), 14336).unwrap();

        assert_eq!(layout.query_heads.len, 32);
        assert_eq!(layout.kv_heads.len, 8);
        assert_eq!(layout.ffn.len, 14336);
    }

    #[test]
    fn a_two_way_split_halves_every_dimension() {
        let m = llama_8b();
        let r0 = ParallelLayout::compute(&m, Rank::new(0, 2).unwrap(), 14336).unwrap();
        let r1 = ParallelLayout::compute(&m, Rank::new(1, 2).unwrap(), 14336).unwrap();

        assert_eq!(r0.query_heads, Shard { start: 0, len: 16 });
        assert_eq!(r1.query_heads, Shard { start: 16, len: 16 });
        assert_eq!(r0.kv_heads.len, 4);
        assert_eq!(r0.ffn.len, 7168);

        // The two ranks together cover the whole model exactly once.
        assert_eq!(r0.query_heads.len + r1.query_heads.len, 32);
        assert_eq!(r0.ffn.end(), r1.ffn.start);
    }

    #[test]
    fn kv_cache_per_gpu_falls_linearly_with_world_size() {
        // The saving that makes tensor parallelism worthwhile for long context.
        let m = llama_8b();
        let single = ParallelLayout::compute(&m, Rank::single(), 14336).unwrap();
        let of_four = ParallelLayout::compute(&m, Rank::new(0, 4).unwrap(), 14336).unwrap();

        let full = single.local_kv_bytes_per_token(&m);
        let quarter = of_four.local_kv_bytes_per_token(&m);

        assert_eq!(full, m.kv_bytes_per_token());
        assert_eq!(quarter * 4, full, "four ranks should each hold a quarter");
    }

    #[test]
    fn splitting_beyond_the_kv_head_count_is_refused() {
        // Llama-3-8B has 8 KV heads, so 16-way tensor parallelism would need
        // them replicated. That is a real limitation, reported rather than
        // silently worked around.
        let m = llama_8b();
        let err = ParallelLayout::compute(&m, Rank::new(0, 16).unwrap(), 14336).unwrap_err();

        assert!(matches!(err, EngineError::InvalidRequest(_)));
        let msg = err.to_string();
        assert!(msg.contains("KV heads"), "{msg}");
        assert!(msg.contains("16"), "{msg}");
    }

    #[test]
    fn eight_way_split_is_the_limit_for_this_model() {
        let m = llama_8b();
        assert!(ParallelLayout::compute(&m, Rank::new(0, 8).unwrap(), 14336).is_ok());
        assert!(ParallelLayout::compute(&m, Rank::new(0, 16).unwrap(), 14336).is_err());
    }

    #[test]
    fn local_attention_width_follows_the_head_split() {
        let m = llama_8b();
        let layout = ParallelLayout::compute(&m, Rank::new(0, 4).unwrap(), 14336).unwrap();
        // 8 heads x 128 dim
        assert_eq!(layout.local_attention_dim(m.head_dim), 1024);
    }

    #[test]
    fn single_rank_collectives_are_identity_operations() {
        let c = SingleRank;
        assert!(!c.rank().is_distributed());

        let mut buf = vec![1.0, 2.0, 3.0];
        c.all_reduce_sum(&mut buf).unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0], "one rank sums to itself");

        let local = vec![4.0, 5.0];
        let mut out = vec![0.0; 2];
        c.all_gather(&local, &mut out).unwrap();
        assert_eq!(out, local);

        let mut scattered = vec![0.0; 2];
        c.reduce_scatter_sum(&local, &mut scattered).unwrap();
        assert_eq!(scattered, local);

        assert!(c.barrier().is_ok());
    }

    #[test]
    fn mismatched_collective_buffers_are_errors() {
        let c = SingleRank;
        let mut out = vec![0.0; 5];
        assert!(c.all_gather(&[1.0, 2.0], &mut out).is_err());
        assert!(c.reduce_scatter_sum(&[1.0, 2.0], &mut out).is_err());
    }

    #[test]
    fn a_single_rank_communicates_nothing() {
        let est = CommunicationEstimate::for_model(&llama_8b(), 1);
        assert_eq!(est.all_reduce_count, 0);
        assert_eq!(est.bytes_per_token, 0);
    }

    #[test]
    fn communication_volume_grows_with_world_size() {
        let m = llama_8b();
        let two = CommunicationEstimate::for_model(&m, 2);
        let eight = CommunicationEstimate::for_model(&m, 8);

        // Two collectives per layer, regardless of world size.
        assert_eq!(two.all_reduce_count, 64);
        assert_eq!(eight.all_reduce_count, 64);

        // But each moves more data as the ring grows.
        assert!(eight.bytes_per_token > two.bytes_per_token);

        // Sanity: 64 AllReduces of a 4096-element f16 vector is on the order of
        // hundreds of KB per token, which is what makes interconnect bandwidth
        // the deciding factor.
        assert!(two.bytes_per_token > 100_000, "{}", two.bytes_per_token);
    }
}
