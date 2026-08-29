//! The abstraction boundaries of the engine.
//!
//! These traits exist to keep three things independent of each other: *what*
//! model is being run, *where* it runs (CPU, CUDA, something else), and *how*
//! requests are ordered. Each boundary is justified below, because an
//! unjustified trait is worse than none — it costs indirection and buys
//! nothing.

use std::fmt::Debug;

use crate::error::EngineError;
use crate::id::{SequenceId, TokenId};
use crate::sampling::SamplingParams;

/// Numeric type of a tensor.
///
/// Deliberately small and closed: the engine only needs to reason about the
/// dtypes it can actually execute, and an open enum would push
/// "unreachable" arms into every backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// 8-bit integer, used by quantized weight formats.
    I8,
    /// 4-bit integer, packed two per byte.
    I4,
}

impl DType {
    /// Bits per element. `I4` is the reason this returns bits and not bytes.
    pub fn bits(self) -> usize {
        match self {
            DType::F32 => 32,
            DType::F16 | DType::BF16 => 16,
            DType::I8 => 8,
            DType::I4 => 4,
        }
    }

    /// Size in bytes of `n` elements of this dtype, rounding up for sub-byte
    /// types.
    pub fn size_in_bytes(self, n: usize) -> usize {
        (n * self.bits()).div_ceil(8)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I8 => "i8",
            DType::I4 => "i4",
        }
    }
}

/// Which device a computation runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// CUDA device with the given ordinal.
    Cuda(usize),
}

impl Device {
    pub fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}

/// Static description of a loaded model, read from `config.json` and validated
/// at load time.
///
/// Everything the scheduler and cache manager need to size themselves is here,
/// so neither has to reach into model internals.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    /// Architecture string from the checkpoint, e.g. `"LlamaForCausalLM"`.
    pub architecture: String,
    /// Human-readable name reported by the API.
    pub name: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads. Equals `num_attention_heads` for multi-head
    /// attention; smaller under grouped-query attention. The KV cache is sized
    /// from *this*, which is exactly why GQA saves so much cache memory.
    pub num_kv_heads: usize,
    /// Dimension per head. Usually `hidden_size / num_attention_heads`, but
    /// stated explicitly because some architectures decouple them.
    pub head_dim: usize,
    pub vocab_size: usize,
    /// Maximum positions the model was trained to handle.
    pub max_position_embeddings: usize,
    /// RoPE base frequency (`theta`).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Weight dtype as stored.
    pub dtype: DType,
    /// EOS token ids. Plural: several chat models define more than one.
    pub eos_token_ids: Vec<TokenId>,
    pub bos_token_id: Option<TokenId>,
}

impl ModelMetadata {
    /// Bytes of KV cache required for a single token, across all layers.
    ///
    /// The factor of two is key and value. This is the number that determines
    /// how many blocks fit in a given memory budget, so it is defined once,
    /// here, rather than recomputed in the cache manager.
    pub fn kv_bytes_per_token(&self) -> usize {
        let elems = 2 * self.num_layers * self.num_kv_heads * self.head_dim;
        self.dtype.size_in_bytes(elems)
    }

    /// Whether the model uses grouped-query attention.
    pub fn uses_gqa(&self) -> bool {
        self.num_kv_heads < self.num_attention_heads
    }

    /// How many query heads share each KV head.
    ///
    /// Returns `0` for a degenerate `num_kv_heads == 0`, which validation
    /// rejects at load time; this is defensive rather than expected.
    pub fn gqa_group_size(&self) -> usize {
        self.num_attention_heads
            .checked_div(self.num_kv_heads)
            .unwrap_or(0)
    }
}

/// A compute backend: where tensors live and where kernels run.
///
/// This boundary exists so the scheduler, cache manager and API can be
/// developed and tested with no GPU present, and so a CUDA backend can be
/// added later without touching them. It is intentionally narrow — it
/// describes the *device*, not the operations, which belong to the model
/// implementation.
pub trait Backend: Send + Sync + Debug {
    /// Stable name for logs and metrics labels.
    fn name(&self) -> &'static str;

    /// The device this backend executes on.
    fn device(&self) -> Device;

    /// Total device memory in bytes, if the backend can report it.
    fn total_memory(&self) -> Option<u64>;

    /// Currently free device memory in bytes, if the backend can report it.
    ///
    /// Used to size the KV cache pool at startup. `None` means the operator
    /// must configure the pool size explicitly.
    fn available_memory(&self) -> Option<u64>;

    /// Blocks until all queued work on the device has completed.
    ///
    /// Required for honest benchmarking: without it, an async launch queue
    /// makes kernels look instantaneous.
    fn synchronize(&self) -> Result<(), EngineError>;

    /// Whether this backend can execute the given dtype natively.
    fn supports_dtype(&self, dtype: DType) -> bool;
}

/// A batch of sequences submitted to the model for one forward pass.
///
/// This is the flattened, "ragged" representation that continuous batching
/// produces: token streams of different sequences concatenated end to end,
/// with offsets describing the boundaries. It avoids padding entirely, which
/// is the whole point — padding to the longest sequence in a mixed
/// prefill/decode batch wastes most of the compute.
#[derive(Debug, Clone)]
pub struct ForwardBatch {
    /// Concatenated input token ids for every sequence in the batch.
    pub tokens: Vec<TokenId>,
    /// Absolute position of each entry in `tokens`, for RoPE.
    pub positions: Vec<u32>,
    /// Sequence id for each slot, parallel to `slot_token_counts`.
    pub sequence_ids: Vec<SequenceId>,
    /// Number of tokens contributed by each sequence. Sums to `tokens.len()`.
    pub slot_token_counts: Vec<usize>,
    /// Context length of each sequence *including* its new tokens. Attention
    /// needs this to know how far back to look in the KV cache.
    pub context_lens: Vec<usize>,
    /// Physical KV block ids per sequence, in logical order.
    pub block_tables: Vec<Vec<u32>>,
    /// True when the batch contains at least one sequence still prefilling.
    /// Lets the backend pick an attention kernel without re-deriving it.
    pub has_prefill: bool,
}

impl ForwardBatch {
    /// Number of sequences in the batch.
    pub fn num_sequences(&self) -> usize {
        self.sequence_ids.len()
    }

    /// Total tokens in the batch — the quantity the scheduler budgets against.
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Checks the parallel-array invariants this struct relies on.
    ///
    /// Called in debug builds and in tests at the scheduler/runtime boundary.
    /// A violation here would otherwise show up as silent numerical garbage
    /// deep inside a kernel.
    pub fn validate(&self) -> Result<(), EngineError> {
        let n = self.sequence_ids.len();
        if self.slot_token_counts.len() != n
            || self.context_lens.len() != n
            || self.block_tables.len() != n
        {
            return Err(EngineError::Internal(format!(
                "ForwardBatch arrays disagree: {n} sequence ids, {} token counts, \
                 {} context lens, {} block tables",
                self.slot_token_counts.len(),
                self.context_lens.len(),
                self.block_tables.len()
            )));
        }
        if self.positions.len() != self.tokens.len() {
            return Err(EngineError::Internal(format!(
                "ForwardBatch has {} tokens but {} positions",
                self.tokens.len(),
                self.positions.len()
            )));
        }
        let summed: usize = self.slot_token_counts.iter().sum();
        if summed != self.tokens.len() {
            return Err(EngineError::Internal(format!(
                "ForwardBatch slot token counts sum to {summed} but there are {} tokens",
                self.tokens.len()
            )));
        }
        Ok(())
    }
}

/// Logits produced by one forward pass.
///
/// Only the *last* position of each sequence is returned, not every position:
/// sampling needs one distribution per sequence, and materializing a
/// `[num_tokens, vocab_size]` tensor for a long prefill would dominate memory
/// traffic for no benefit.
#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Row-major `[num_sequences, vocab_size]` logits.
    pub logits: Vec<f32>,
    pub vocab_size: usize,
    /// Sequence ids, parallel to the rows of `logits`.
    pub sequence_ids: Vec<SequenceId>,
}

impl ForwardOutput {
    /// Borrows the logits row belonging to the `i`-th sequence.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        let start = i.checked_mul(self.vocab_size)?;
        let end = start.checked_add(self.vocab_size)?;
        self.logits.get(start..end)
    }

    pub fn num_sequences(&self) -> usize {
        self.sequence_ids.len()
    }
}

/// A decoder-only language model that can run a batched forward pass.
///
/// `&self` rather than `&mut self`: model weights are immutable after loading,
/// and all mutable state (the KV cache) lives outside the model. That is what
/// makes it sound to share one model across worker threads.
pub trait LanguageModel: Send + Sync {
    /// Static description of the model.
    fn metadata(&self) -> &ModelMetadata;

    /// Runs one forward pass over a ragged batch.
    ///
    /// Reads and writes KV entries through the block tables carried in
    /// `batch`. Returns last-position logits for every sequence.
    fn forward(&self, batch: &ForwardBatch) -> Result<ForwardOutput, EngineError>;

    /// The backend this model executes on.
    fn backend(&self) -> &dyn Backend;
}

/// Turns a logits row into a chosen token.
///
/// Separated from model execution so it can be unit-tested against known
/// distributions with no model present, and so seeded runs are reproducible
/// independently of how the logits were computed.
pub trait Sampler: Send + Sync {
    /// Picks the next token.
    ///
    /// `previous_tokens` is the full context, needed for repetition penalty.
    /// `logits` is mutated in place — the caller owns a scratch row, and
    /// copying a vocabulary-sized buffer per sequence per step is exactly the
    /// kind of allocation the decode loop cannot afford.
    fn sample(
        &mut self,
        logits: &mut [f32],
        previous_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<TokenId, EngineError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "LlamaForCausalLM".into(),
            name: "test".into(),
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            dtype: DType::F16,
            eos_token_ids: vec![2],
            bos_token_id: Some(1),
        }
    }

    #[test]
    fn sub_byte_dtypes_round_up_to_whole_bytes() {
        assert_eq!(DType::I4.size_in_bytes(1), 1);
        assert_eq!(DType::I4.size_in_bytes(2), 1);
        assert_eq!(DType::I4.size_in_bytes(3), 2);
        assert_eq!(DType::F16.size_in_bytes(4), 8);
        assert_eq!(DType::F32.size_in_bytes(4), 16);
    }

    #[test]
    fn kv_bytes_per_token_accounts_for_key_and_value() {
        let m = metadata();
        // 2 (K+V) * 32 layers * 8 kv heads * 128 dim * 2 bytes (f16)
        assert_eq!(m.kv_bytes_per_token(), 2 * 32 * 8 * 128 * 2);
    }

    #[test]
    fn gqa_is_detected_and_grouped_correctly() {
        let m = metadata();
        assert!(m.uses_gqa());
        assert_eq!(m.gqa_group_size(), 4);

        let mha = ModelMetadata {
            num_kv_heads: 32,
            ..metadata()
        };
        assert!(!mha.uses_gqa());
        assert_eq!(mha.gqa_group_size(), 1);
        // MHA needs 4x the cache of the GQA variant.
        assert_eq!(
            mha.kv_bytes_per_token(),
            4 * metadata().kv_bytes_per_token()
        );
    }

    #[test]
    fn device_displays_with_its_ordinal() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cuda(0).is_cpu());
    }

    fn batch() -> ForwardBatch {
        ForwardBatch {
            tokens: vec![1, 2, 3, 4],
            positions: vec![0, 1, 2, 0],
            sequence_ids: vec![SequenceId::from_raw(1), SequenceId::from_raw(2)],
            slot_token_counts: vec![3, 1],
            context_lens: vec![3, 1],
            block_tables: vec![vec![0], vec![1]],
            has_prefill: true,
        }
    }

    #[test]
    fn a_consistent_batch_validates() {
        let b = batch();
        assert!(b.validate().is_ok());
        assert_eq!(b.num_sequences(), 2);
        assert_eq!(b.num_tokens(), 4);
    }

    #[test]
    fn batch_validation_catches_mismatched_parallel_arrays() {
        let mut b = batch();
        b.context_lens.pop();
        assert!(b.validate().is_err());

        let mut b = batch();
        b.positions.pop();
        assert!(b.validate().is_err());

        let mut b = batch();
        b.slot_token_counts = vec![2, 1]; // sums to 3, not 4
        assert!(b.validate().is_err());
    }

    #[test]
    fn forward_output_rows_are_bounds_checked() {
        let out = ForwardOutput {
            logits: vec![0.0; 6],
            vocab_size: 3,
            sequence_ids: vec![SequenceId::from_raw(1), SequenceId::from_raw(2)],
        };
        assert_eq!(out.row(0).unwrap().len(), 3);
        assert_eq!(out.row(1).unwrap().len(), 3);
        assert!(out.row(2).is_none());
    }
}
