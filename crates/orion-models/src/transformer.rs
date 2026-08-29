//! The decoder-only transformer forward pass.
//!
//! Assembles the pieces — embedding, RMSNorm, RoPE, paged attention, SwiGLU
//! MLP, LM head — into a [`LanguageModel`] the engine can drive.
//!
//! # Weight naming
//!
//! Tensor names follow the Hugging Face Llama convention, which Qwen2 also
//! uses:
//!
//! ```text
//! model.embed_tokens.weight
//! model.layers.{i}.input_layernorm.weight
//! model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//! model.layers.{i}.post_attention_layernorm.weight
//! model.layers.{i}.mlp.{gate,up,down}_proj.weight
//! model.norm.weight
//! lm_head.weight
//! ```

use std::sync::Mutex;

use orion_core::{
    Backend, EngineError, ForwardBatch, ForwardOutput, LanguageModel, ModelError, ModelMetadata,
    SequenceId,
};

use crate::attention::{locate, paged_attention, AttentionArgs, KvStore};
use crate::config::{Architecture, HfConfig};
use crate::loader::CheckpointLoader;
use crate::tensor::{apply_rope, linear, rms_norm, swiglu, Matrix, RopeTable};

/// Weights for one transformer block.
#[derive(Debug)]
pub struct LayerWeights {
    pub input_norm: Vec<f32>,
    pub q_proj: Matrix,
    pub k_proj: Matrix,
    pub v_proj: Matrix,
    pub o_proj: Matrix,
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    pub post_attn_norm: Vec<f32>,
    pub gate_proj: Matrix,
    pub up_proj: Matrix,
    pub down_proj: Matrix,
}

/// All weights of a loaded model.
#[derive(Debug)]
pub struct ModelWeights {
    pub embed_tokens: Matrix,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    /// `None` when the LM head is tied to the embedding matrix.
    pub lm_head: Option<Matrix>,
}

impl ModelWeights {
    /// Loads every weight named by the Llama/Qwen2 convention.
    pub fn load(
        loader: &CheckpointLoader,
        config: &HfConfig,
        meta: &ModelMetadata,
        arch: Architecture,
    ) -> Result<Self, ModelError> {
        let hidden = meta.hidden_size;
        let ffn = config.ffn_dim();
        let q_dim = meta.num_attention_heads * meta.head_dim;
        let kv_dim = meta.num_kv_heads * meta.head_dim;

        let embed_tokens = loader.matrix("model.embed_tokens.weight", meta.vocab_size, hidden)?;

        let mut layers = Vec::with_capacity(meta.num_layers);
        for i in 0..meta.num_layers {
            let p = format!("model.layers.{i}");
            let (q_bias, k_bias, v_bias) = if arch.has_qkv_bias() {
                (
                    loader.optional_vector(&format!("{p}.self_attn.q_proj.bias"), q_dim)?,
                    loader.optional_vector(&format!("{p}.self_attn.k_proj.bias"), kv_dim)?,
                    loader.optional_vector(&format!("{p}.self_attn.v_proj.bias"), kv_dim)?,
                )
            } else {
                (None, None, None)
            };

            layers.push(LayerWeights {
                input_norm: loader.vector(&format!("{p}.input_layernorm.weight"), hidden)?,
                q_proj: loader.matrix(&format!("{p}.self_attn.q_proj.weight"), q_dim, hidden)?,
                k_proj: loader.matrix(&format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden)?,
                v_proj: loader.matrix(&format!("{p}.self_attn.v_proj.weight"), kv_dim, hidden)?,
                o_proj: loader.matrix(&format!("{p}.self_attn.o_proj.weight"), hidden, q_dim)?,
                q_bias,
                k_bias,
                v_bias,
                post_attn_norm: loader
                    .vector(&format!("{p}.post_attention_layernorm.weight"), hidden)?,
                gate_proj: loader.matrix(&format!("{p}.mlp.gate_proj.weight"), ffn, hidden)?,
                up_proj: loader.matrix(&format!("{p}.mlp.up_proj.weight"), ffn, hidden)?,
                down_proj: loader.matrix(&format!("{p}.mlp.down_proj.weight"), hidden, ffn)?,
            });
        }

        // A tied head reuses the embedding matrix rather than storing a second
        // copy of a vocab_size x hidden matrix, which for a 128k vocabulary is
        // a substantial saving.
        let lm_head = if config.ties_embeddings() || !loader.contains("lm_head.weight") {
            None
        } else {
            Some(loader.matrix("lm_head.weight", meta.vocab_size, hidden)?)
        };

        Ok(Self {
            embed_tokens,
            layers,
            final_norm: loader.vector("model.norm.weight", hidden)?,
            lm_head,
        })
    }
}

/// A CPU-executing decoder-only transformer.
///
/// # Interior mutability
///
/// [`LanguageModel::forward`] takes `&self` so one model can be shared across
/// threads without synchronizing the weights, which are immutable. The KV cache
/// is the sole mutable state, and it sits behind a `Mutex` here.
///
/// The mutex is uncontended in practice: the engine step loop is
/// single-threaded by design (ADR 006), so there is exactly one caller. It
/// exists to satisfy `Sync` for the shared-model case rather than to arbitrate
/// real contention, and it is documented as such so a future reader does not
/// mistake it for a hot lock.
#[derive(Debug)]
pub struct TransformerModel {
    meta: ModelMetadata,
    weights: ModelWeights,
    rope: RopeTable,
    kv: Mutex<KvStore>,
    block_size: usize,
    backend: Box<dyn Backend>,
}

impl TransformerModel {
    /// Assembles a model from loaded weights.
    pub fn new(
        meta: ModelMetadata,
        weights: ModelWeights,
        num_blocks: usize,
        block_size: usize,
        backend: Box<dyn Backend>,
    ) -> Result<Self, ModelError> {
        let rope = RopeTable::new(meta.head_dim, meta.max_position_embeddings, meta.rope_theta)
            .map_err(|e| ModelError::Malformed {
                file: "config.json".into(),
                reason: e.to_string(),
            })?;

        let kv = KvStore::new(
            meta.num_layers,
            num_blocks,
            block_size,
            meta.num_kv_heads,
            meta.head_dim,
        );

        Ok(Self {
            meta,
            weights,
            rope,
            kv: Mutex::new(kv),
            block_size,
            backend,
        })
    }

    /// Loads a model from a Hugging Face directory.
    pub fn from_directory(
        dir: &std::path::Path,
        num_blocks: usize,
        block_size: usize,
        backend: Box<dyn Backend>,
    ) -> Result<Self, ModelError> {
        let config = HfConfig::load(dir)?;
        let arch = Architecture::detect(&config)?;
        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into());
        let meta = config.to_metadata(name)?;

        let loader = CheckpointLoader::open(dir)?;
        let weights = ModelWeights::load(&loader, &config, &meta, arch)?;

        tracing::info!(
            architecture = %meta.architecture,
            layers = meta.num_layers,
            hidden = meta.hidden_size,
            kv_heads = meta.num_kv_heads,
            "loaded model"
        );
        Self::new(meta, weights, num_blocks, block_size, backend)
    }

    /// Looks up the embedding rows for a batch of token ids.
    fn embed(&self, tokens: &[u32]) -> Result<Matrix, EngineError> {
        let hidden = self.meta.hidden_size;
        let mut out = Matrix::zeros(tokens.len(), hidden);
        for (i, &tok) in tokens.iter().enumerate() {
            let row = self.weights.embed_tokens.row(tok as usize).ok_or_else(|| {
                EngineError::Internal(format!(
                    "token id {tok} is outside the vocabulary of {}",
                    self.meta.vocab_size
                ))
            })?;
            out.row_mut(i)
                .expect("row index is in range by construction")
                .copy_from_slice(row);
        }
        Ok(out)
    }

    /// Runs one transformer block over the batch.
    ///
    /// `positions` gives the absolute position of each row, and `slots` the
    /// physical (block, slot) each row's KV entry is written to.
    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer_idx: usize,
        hidden: &mut Matrix,
        positions: &[u32],
        slots: &[(u32, usize)],
        seq_of_row: &[usize],
        block_tables: &[Vec<u32>],
        context_lens: &[usize],
    ) -> Result<(), EngineError> {
        let w = &self.weights.layers[layer_idx];
        let meta = &self.meta;
        let n = hidden.rows();

        // --- Attention block, pre-norm ---
        let mut normed = hidden.clone();
        rms_norm(&mut normed, &w.input_norm, meta.rms_norm_eps)?;

        let mut q = linear(&normed, &w.q_proj, w.q_bias.as_deref())?;
        let mut k = linear(&normed, &w.k_proj, w.k_bias.as_deref())?;
        let v = linear(&normed, &w.v_proj, w.v_bias.as_deref())?;

        // Rotary embedding is applied to Q and K only. V carries no positional
        // information, which is what lets a cached V stay valid regardless of
        // where it is later attended from.
        for (row, &position) in positions.iter().enumerate().take(n) {
            let pos = position as usize;
            let (cos, sin) = self.rope.at(pos).ok_or_else(|| {
                EngineError::Internal(format!(
                    "position {pos} exceeds the RoPE table ({} entries)",
                    self.rope.max_positions()
                ))
            })?;
            let qrow = q.row_mut(row).expect("row in range");
            for h in 0..meta.num_attention_heads {
                apply_rope(
                    &mut qrow[h * meta.head_dim..(h + 1) * meta.head_dim],
                    cos,
                    sin,
                )?;
            }
            let krow = k.row_mut(row).expect("row in range");
            for h in 0..meta.num_kv_heads {
                apply_rope(
                    &mut krow[h * meta.head_dim..(h + 1) * meta.head_dim],
                    cos,
                    sin,
                )?;
            }
        }

        // Write this step's K and V into the paged cache before attending, so
        // a token can attend to itself.
        {
            let mut kv = self
                .kv
                .lock()
                .map_err(|_| EngineError::Internal("KV cache mutex poisoned".into()))?;
            for (row, &(block, slot)) in slots.iter().enumerate().take(n) {
                let krow = k.row(row).expect("row in range");
                let vrow = v.row(row).expect("row in range");
                for h in 0..meta.num_kv_heads {
                    let span = h * meta.head_dim..(h + 1) * meta.head_dim;
                    kv.write(layer_idx, block, slot, h, false, &krow[span.clone()])?;
                    kv.write(layer_idx, block, slot, h, true, &vrow[span])?;
                }
            }
        }

        // Attention, one row at a time.
        let mut attn_out = Matrix::zeros(n, meta.num_attention_heads * meta.head_dim);
        {
            let kv = self
                .kv
                .lock()
                .map_err(|_| EngineError::Internal("KV cache mutex poisoned".into()))?;
            for row in 0..n {
                let seq = seq_of_row[row];
                let qrow = q.row(row).expect("row in range");
                let orow = attn_out.row_mut(row).expect("row in range");
                paged_attention(
                    &kv,
                    qrow,
                    AttentionArgs {
                        block_table: &block_tables[seq],
                        context_len: context_lens[seq],
                        query_position: positions[row] as usize,
                        layer: layer_idx,
                        num_query_heads: meta.num_attention_heads,
                        num_kv_heads: meta.num_kv_heads,
                        head_dim: meta.head_dim,
                        block_size: self.block_size,
                    },
                    orow,
                )?;
            }
        }

        let projected = linear(&attn_out, &w.o_proj, None)?;
        // Residual connection.
        for (h, p) in hidden.data_mut().iter_mut().zip(projected.data().iter()) {
            *h += p;
        }

        // --- MLP block, pre-norm ---
        let mut normed = hidden.clone();
        rms_norm(&mut normed, &w.post_attn_norm, meta.rms_norm_eps)?;

        let mut gate = linear(&normed, &w.gate_proj, None)?;
        let up = linear(&normed, &w.up_proj, None)?;
        swiglu(&mut gate, &up)?;
        let mlp_out = linear(&gate, &w.down_proj, None)?;

        for (h, m) in hidden.data_mut().iter_mut().zip(mlp_out.data().iter()) {
            *h += m;
        }
        Ok(())
    }
}

impl LanguageModel for TransformerModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }

    fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    fn forward(&self, batch: &ForwardBatch) -> Result<ForwardOutput, EngineError> {
        batch.validate()?;
        if batch.is_empty() {
            return Ok(ForwardOutput {
                logits: Vec::new(),
                vocab_size: self.meta.vocab_size,
                sequence_ids: Vec::new(),
            });
        }

        // Map each flat row back to the sequence that contributed it, and
        // resolve where its KV entry goes.
        let mut seq_of_row = Vec::with_capacity(batch.tokens.len());
        let mut slots = Vec::with_capacity(batch.tokens.len());
        for (seq_idx, &count) in batch.slot_token_counts.iter().enumerate() {
            for _ in 0..count {
                seq_of_row.push(seq_idx);
            }
        }
        for (row, &pos) in batch.positions.iter().enumerate() {
            let seq = seq_of_row[row];
            let (block, slot) = locate(&batch.block_tables[seq], pos as usize, self.block_size)
                .ok_or_else(|| {
                    EngineError::Internal(format!(
                        "position {pos} has no block in the table for sequence {seq}"
                    ))
                })?;
            slots.push((block, slot));
        }

        let mut hidden = self.embed(&batch.tokens)?;
        for layer in 0..self.meta.num_layers {
            self.layer_forward(
                layer,
                &mut hidden,
                &batch.positions,
                &slots,
                &seq_of_row,
                &batch.block_tables,
                &batch.context_lens,
            )?;
        }
        rms_norm(
            &mut hidden,
            &self.weights.final_norm,
            self.meta.rms_norm_eps,
        )?;

        // Only the last row of each sequence needs logits. Computing them for
        // every position of a long prefill would multiply the LM head cost by
        // the prompt length for output nobody reads.
        let mut last_rows = Vec::with_capacity(batch.num_sequences());
        let mut cursor = 0usize;
        for &count in &batch.slot_token_counts {
            if count == 0 {
                return Err(EngineError::Internal(
                    "a scheduled sequence contributed zero tokens".into(),
                ));
            }
            cursor += count;
            last_rows.push(cursor - 1);
        }

        let mut tail = Matrix::zeros(last_rows.len(), self.meta.hidden_size);
        for (i, &row) in last_rows.iter().enumerate() {
            let src = hidden.row(row).ok_or_else(|| {
                EngineError::Internal(format!("row {row} missing from hidden states"))
            })?;
            tail.row_mut(i).expect("row in range").copy_from_slice(src);
        }

        let head = self
            .weights
            .lm_head
            .as_ref()
            .unwrap_or(&self.weights.embed_tokens);
        let logits = linear(&tail, head, None)?;

        Ok(ForwardOutput {
            logits: logits.into_data(),
            vocab_size: self.meta.vocab_size,
            sequence_ids: batch.sequence_ids.clone(),
        })
    }
}

/// Sequence-id helper used by tests and by the engine when constructing
/// batches by hand.
pub fn sequence_ids(n: usize) -> Vec<SequenceId> {
    (1..=n as u64).map(SequenceId::from_raw).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use orion_core::DType;

    /// A tiny model with deterministic weights, so the forward pass can be
    /// checked end to end without a real checkpoint.
    fn tiny_meta(num_layers: usize) -> ModelMetadata {
        ModelMetadata {
            architecture: "llama".into(),
            name: "tiny".into(),
            hidden_size: 4,
            num_layers,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            vocab_size: 8,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            dtype: DType::F32,
            eos_token_ids: vec![7],
            bos_token_id: Some(0),
        }
    }

    /// Deterministic pseudo-random weights: small, varied, reproducible.
    fn weight_matrix(rows: usize, cols: usize, seed: f32) -> Matrix {
        let data = (0..rows * cols)
            .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
            .collect();
        Matrix::new(data, rows, cols).unwrap()
    }

    fn tiny_weights(meta: &ModelMetadata, ffn: usize) -> ModelWeights {
        let h = meta.hidden_size;
        let q_dim = meta.num_attention_heads * meta.head_dim;
        let kv_dim = meta.num_kv_heads * meta.head_dim;

        let layers = (0..meta.num_layers)
            .map(|i| {
                let s = i as f32;
                LayerWeights {
                    input_norm: vec![1.0; h],
                    q_proj: weight_matrix(q_dim, h, s + 1.0),
                    k_proj: weight_matrix(kv_dim, h, s + 2.0),
                    v_proj: weight_matrix(kv_dim, h, s + 3.0),
                    o_proj: weight_matrix(h, q_dim, s + 4.0),
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,
                    post_attn_norm: vec![1.0; h],
                    gate_proj: weight_matrix(ffn, h, s + 5.0),
                    up_proj: weight_matrix(ffn, h, s + 6.0),
                    down_proj: weight_matrix(h, ffn, s + 7.0),
                }
            })
            .collect();

        ModelWeights {
            embed_tokens: weight_matrix(meta.vocab_size, h, 0.5),
            layers,
            final_norm: vec![1.0; h],
            lm_head: None,
        }
    }

    fn tiny_model(num_layers: usize) -> TransformerModel {
        let meta = tiny_meta(num_layers);
        let weights = tiny_weights(&meta, 8);
        TransformerModel::new(meta, weights, 16, 4, Box::new(CpuBackend::new())).unwrap()
    }

    fn batch_for(tokens: Vec<u32>, positions: Vec<u32>, blocks: Vec<u32>) -> ForwardBatch {
        let n = tokens.len();
        ForwardBatch {
            tokens,
            positions,
            sequence_ids: sequence_ids(1),
            slot_token_counts: vec![n],
            context_lens: vec![n],
            block_tables: vec![blocks],
            has_prefill: true,
        }
    }

    #[test]
    fn a_forward_pass_produces_finite_logits_of_the_right_shape() {
        let m = tiny_model(2);
        let batch = batch_for(vec![1, 2, 3], vec![0, 1, 2], vec![0]);
        let out = m.forward(&batch).unwrap();

        assert_eq!(out.vocab_size, 8);
        assert_eq!(out.num_sequences(), 1);
        assert_eq!(out.logits.len(), 8, "one row of logits per sequence");
        assert!(
            out.logits.iter().all(|l| l.is_finite()),
            "non-finite logits: {:?}",
            out.logits
        );
    }

    #[test]
    fn only_the_last_position_yields_logits() {
        // A 5-token prefill must still produce exactly one logits row.
        let m = tiny_model(1);
        let batch = batch_for(vec![1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4], vec![0, 1]);
        let out = m.forward(&batch).unwrap();
        assert_eq!(out.logits.len(), 8);
    }

    #[test]
    fn an_empty_batch_returns_empty_output() {
        let m = tiny_model(1);
        let batch = ForwardBatch {
            tokens: vec![],
            positions: vec![],
            sequence_ids: vec![],
            slot_token_counts: vec![],
            context_lens: vec![],
            block_tables: vec![],
            has_prefill: false,
        };
        let out = m.forward(&batch).unwrap();
        assert!(out.logits.is_empty());
        assert_eq!(out.num_sequences(), 0);
    }

    #[test]
    fn a_batch_of_several_sequences_yields_one_row_each() {
        let m = tiny_model(1);
        let batch = ForwardBatch {
            tokens: vec![1, 2, 3, 4],
            positions: vec![0, 1, 0, 1],
            sequence_ids: sequence_ids(2),
            slot_token_counts: vec![2, 2],
            context_lens: vec![2, 2],
            block_tables: vec![vec![0], vec![1]],
            has_prefill: true,
        };
        let out = m.forward(&batch).unwrap();

        assert_eq!(out.num_sequences(), 2);
        assert_eq!(out.logits.len(), 16, "2 sequences x 8 vocab");
        assert!(out.row(0).is_some() && out.row(1).is_some());
        assert!(out.row(2).is_none());
    }

    #[test]
    fn batched_sequences_do_not_influence_each_other() {
        // The property that makes continuous batching sound: a sequence's
        // logits must not depend on what else was in the batch.
        let m = tiny_model(2);

        let alone = m
            .forward(&ForwardBatch {
                tokens: vec![1, 2],
                positions: vec![0, 1],
                sequence_ids: sequence_ids(1),
                slot_token_counts: vec![2],
                context_lens: vec![2],
                block_tables: vec![vec![0]],
                has_prefill: true,
            })
            .unwrap();

        let together = m
            .forward(&ForwardBatch {
                tokens: vec![1, 2, 5, 6, 7],
                positions: vec![0, 1, 0, 1, 2],
                sequence_ids: sequence_ids(2),
                slot_token_counts: vec![2, 3],
                context_lens: vec![2, 3],
                block_tables: vec![vec![0], vec![4]],
                has_prefill: true,
            })
            .unwrap();

        let a = alone.row(0).unwrap();
        let b = together.row(0).unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "batching changed the result: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn incremental_decode_matches_a_full_prefill() {
        // The property that makes the KV cache sound: feeding tokens one at a
        // time must give the same answer as processing them all at once.
        let full = {
            let m = tiny_model(2);
            let out = m
                .forward(&batch_for(vec![1, 2, 3], vec![0, 1, 2], vec![0]))
                .unwrap();
            out.row(0).unwrap().to_vec()
        };

        let incremental = {
            let m = tiny_model(2);
            // Prefill the first two, then decode the third.
            m.forward(&batch_for(vec![1, 2], vec![0, 1], vec![0]))
                .unwrap();
            let out = m
                .forward(&ForwardBatch {
                    tokens: vec![3],
                    positions: vec![2],
                    sequence_ids: sequence_ids(1),
                    slot_token_counts: vec![1],
                    context_lens: vec![3],
                    block_tables: vec![vec![0]],
                    has_prefill: false,
                })
                .unwrap();
            out.row(0).unwrap().to_vec()
        };

        for (a, b) in full.iter().zip(incremental.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "cached decode diverged from full prefill:\n{full:?}\n{incremental:?}"
            );
        }
    }

    #[test]
    fn chunked_prefill_matches_a_single_pass() {
        // Chunking a prompt across steps must not change the result.
        let whole = {
            let m = tiny_model(2);
            m.forward(&batch_for(vec![1, 2, 3, 4], vec![0, 1, 2, 3], vec![0]))
                .unwrap()
                .row(0)
                .unwrap()
                .to_vec()
        };

        let chunked = {
            let m = tiny_model(2);
            m.forward(&batch_for(vec![1, 2], vec![0, 1], vec![0]))
                .unwrap();
            m.forward(&ForwardBatch {
                tokens: vec![3, 4],
                positions: vec![2, 3],
                sequence_ids: sequence_ids(1),
                slot_token_counts: vec![2],
                context_lens: vec![4],
                block_tables: vec![vec![0]],
                has_prefill: true,
            })
            .unwrap()
            .row(0)
            .unwrap()
            .to_vec()
        };

        for (a, b) in whole.iter().zip(chunked.iter()) {
            assert!((a - b).abs() < 1e-4, "{whole:?} vs {chunked:?}");
        }
    }

    #[test]
    fn a_scattered_block_table_gives_identical_logits() {
        // Paging must be invisible to the arithmetic.
        let contiguous = {
            let m = tiny_model(2);
            m.forward(&batch_for(
                vec![1, 2, 3, 4, 5],
                vec![0, 1, 2, 3, 4],
                vec![0, 1],
            ))
            .unwrap()
            .row(0)
            .unwrap()
            .to_vec()
        };
        let scattered = {
            let m = tiny_model(2);
            m.forward(&batch_for(
                vec![1, 2, 3, 4, 5],
                vec![0, 1, 2, 3, 4],
                vec![11, 3],
            ))
            .unwrap()
            .row(0)
            .unwrap()
            .to_vec()
        };

        for (a, b) in contiguous.iter().zip(scattered.iter()) {
            assert!((a - b).abs() < 1e-5, "{contiguous:?} vs {scattered:?}");
        }
    }

    #[test]
    fn different_prompts_give_different_logits() {
        // A sanity check that the model is actually reading its input.
        let m = tiny_model(2);
        let a = m
            .forward(&batch_for(vec![1, 1, 1], vec![0, 1, 2], vec![0]))
            .unwrap()
            .row(0)
            .unwrap()
            .to_vec();
        let b = m
            .forward(&batch_for(vec![5, 6, 4], vec![0, 1, 2], vec![2]))
            .unwrap()
            .row(0)
            .unwrap()
            .to_vec();

        let differs = a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4);
        assert!(differs, "the model ignored its input");
    }

    #[test]
    fn an_out_of_vocabulary_token_is_an_error_not_a_panic() {
        let m = tiny_model(1);
        let err = m
            .forward(&batch_for(vec![999], vec![0], vec![0]))
            .unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
    }

    #[test]
    fn a_position_beyond_the_rope_table_is_an_error() {
        let m = tiny_model(1);
        // max_position_embeddings is 32.
        let err = m
            .forward(&batch_for(
                vec![1],
                vec![100],
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25,
                ],
            ))
            .unwrap_err();
        assert!(matches!(err, EngineError::Internal(_)));
    }

    #[test]
    fn an_inconsistent_batch_is_rejected_before_compute() {
        let m = tiny_model(1);
        let mut batch = batch_for(vec![1, 2], vec![0, 1], vec![0]);
        batch.slot_token_counts = vec![5]; // does not match token count
        assert!(m.forward(&batch).is_err());
    }

    #[test]
    fn metadata_is_reported_and_the_backend_is_cpu() {
        let m = tiny_model(1);
        assert_eq!(m.metadata().num_layers, 1);
        assert_eq!(m.backend().name(), "cpu");
        assert!(m.backend().device().is_cpu());
    }
}
