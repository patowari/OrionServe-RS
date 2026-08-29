//! The engine: the loop that turns scheduled batches into generated tokens.
//!
//! # Threading
//!
//! The step loop runs on **one** dedicated thread (ADR 006). Async request
//! handlers submit over a bounded channel and receive tokens back over
//! per-request channels. The scheduler and cache manager are therefore owned
//! outright by the loop and mutated through `&mut self`, with no locking.
//!
//! Bounded channels are what provide backpressure: when the engine falls
//! behind, submission fails rather than queueing without limit, and the API
//! layer converts that into a 429.

use std::sync::Arc;

use orion_core::{
    EngineError, FinishReason, ForwardBatch, LanguageModel, Sampler, SamplingParams, Sequence,
    SequenceId, TokenId,
};
use orion_kv_cache::KvCacheManager;
use orion_scheduler::{Scheduler, SchedulerStats};
use tokio::sync::{mpsc, oneshot};

use crate::sampling::DefaultSampler;

/// One unit of output streamed back to a caller.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A generated token, with the text it contributes.
    ///
    /// `text` may be empty when the token completes no character yet; callers
    /// forward it as a zero-length delta or skip it.
    Token {
        token: TokenId,
        text: String,
        /// Total tokens generated so far, for usage accounting.
        index: usize,
    },
    /// Generation finished.
    Done {
        reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// Generation failed. Terminal.
    ///
    /// Carries the stable error code and message rather than the
    /// [`EngineError`] itself: `EngineError` is deliberately not `Clone`
    /// (it wraps `io::Error`), while a stream event must be, and the API layer
    /// needs exactly these two fields to build a response body.
    Error { code: &'static str, message: String },
}

impl StreamEvent {
    /// Builds an error event from an engine error.
    pub fn from_error(e: &EngineError) -> Self {
        StreamEvent::Error {
            code: e.code(),
            message: e.to_string(),
        }
    }

    /// Whether this event ends the stream.
    pub fn is_terminal(&self) -> bool {
        matches!(self, StreamEvent::Done { .. } | StreamEvent::Error { .. })
    }
}

/// A request handed to the engine.
#[derive(Debug)]
pub struct GenerationRequest {
    pub prompt: Vec<TokenId>,
    pub params: SamplingParams,
    /// Where output events go.
    pub events: mpsc::Sender<StreamEvent>,
    /// Resolves with the assigned sequence id, or the rejection reason.
    pub accepted: oneshot::Sender<Result<SequenceId, EngineError>>,
}

/// Messages the engine thread accepts.
#[derive(Debug)]
enum Command {
    Generate(Box<GenerationRequest>),
    Cancel(SequenceId),
    /// Requests a stats snapshot.
    Stats(oneshot::Sender<EngineStats>),
    Shutdown,
}

/// A snapshot of engine state, for `/metrics` and health checks.
#[derive(Debug, Clone, Copy, Default)]
pub struct EngineStats {
    pub scheduler: SchedulerStats,
    pub running: usize,
    pub waiting: usize,
    pub cache_total_blocks: usize,
    pub cache_free_blocks: usize,
    pub prefix_cache_hits: u64,
    pub prefix_cache_misses: u64,
}

impl EngineStats {
    pub fn cache_utilization(&self) -> f64 {
        if self.cache_total_blocks == 0 {
            0.0
        } else {
            let used = self.cache_total_blocks - self.cache_free_blocks;
            used as f64 / self.cache_total_blocks as f64
        }
    }

    pub fn prefix_cache_hit_rate(&self) -> f64 {
        let total = self.prefix_cache_hits + self.prefix_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.prefix_cache_hits as f64 / total as f64
        }
    }
}

/// Handle used by the API layer to talk to the engine thread.
#[derive(Debug, Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Command>,
}

impl EngineHandle {
    /// Submits a request and waits for admission.
    ///
    /// Returns the sequence id once accepted. Rejection — a full queue, an
    /// oversized context — comes back as an error here rather than as a stream
    /// event, so the API layer can answer with a status code before opening a
    /// response body.
    pub async fn generate(
        &self,
        prompt: Vec<TokenId>,
        params: SamplingParams,
        events: mpsc::Sender<StreamEvent>,
    ) -> Result<SequenceId, EngineError> {
        let (accepted_tx, accepted_rx) = oneshot::channel();
        let req = GenerationRequest {
            prompt,
            params,
            events,
            accepted: accepted_tx,
        };

        self.tx
            .send(Command::Generate(Box::new(req)))
            .await
            .map_err(|_| EngineError::EngineShutdown)?;

        accepted_rx.await.map_err(|_| EngineError::EngineShutdown)?
    }

    /// Cancels a request. Safe to call for one that already finished.
    pub async fn cancel(&self, id: SequenceId) {
        let _ = self.tx.send(Command::Cancel(id)).await;
    }

    /// Reads current engine statistics.
    pub async fn stats(&self) -> Result<EngineStats, EngineError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Stats(tx))
            .await
            .map_err(|_| EngineError::EngineShutdown)?;
        rx.await.map_err(|_| EngineError::EngineShutdown)
    }

    /// Asks the engine to stop after draining in-flight work.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(Command::Shutdown).await;
    }

    /// Whether the engine thread is still accepting commands.
    pub fn is_alive(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// Per-sequence state the engine tracks alongside the scheduler's own.
struct ActiveRequest {
    events: mpsc::Sender<StreamEvent>,
    sampler: DefaultSampler,
    prompt_len: usize,
}

/// Everything the engine loop owns.
///
/// Not `Clone` and not shared: exactly one of these exists, on the engine
/// thread. Callers interact through [`EngineHandle`].
pub struct Engine {
    scheduler: Scheduler<KvCacheManager>,
    model: Arc<dyn LanguageModel>,
    active: std::collections::HashMap<SequenceId, ActiveRequest>,
    eos_ids: Vec<TokenId>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("running", &self.scheduler.num_running())
            .field("waiting", &self.scheduler.num_waiting())
            .field("active", &self.active.len())
            .finish()
    }
}

impl Engine {
    pub fn new(scheduler: Scheduler<KvCacheManager>, model: Arc<dyn LanguageModel>) -> Self {
        let eos_ids = model.metadata().eos_token_ids.clone();
        Self {
            scheduler,
            model,
            active: std::collections::HashMap::new(),
            eos_ids,
        }
    }

    pub fn stats(&self) -> EngineStats {
        let cache = self.scheduler.cache().stats();
        EngineStats {
            scheduler: self.scheduler.stats(),
            running: self.scheduler.num_running(),
            waiting: self.scheduler.num_waiting(),
            cache_total_blocks: cache.total_blocks,
            cache_free_blocks: cache.free_blocks,
            prefix_cache_hits: cache.prefix_cache_hits,
            prefix_cache_misses: cache.prefix_cache_misses,
        }
    }

    /// Admits a request, registering its output channel.
    pub fn submit(&mut self, req: GenerationRequest) {
        let GenerationRequest {
            prompt,
            params,
            events,
            accepted,
        } = req;

        if let Err(e) = params.validate() {
            let _ = accepted.send(Err(e));
            return;
        }

        let prompt_len = prompt.len();
        let sampler = DefaultSampler::for_params(&params);
        let seq = Sequence::new(prompt, params);

        match self.scheduler.add_request(seq) {
            Ok(id) => {
                self.active.insert(
                    id,
                    ActiveRequest {
                        events,
                        sampler,
                        prompt_len,
                    },
                );
                let _ = accepted.send(Ok(id));
            }
            Err(e) => {
                let _ = accepted.send(Err(e));
            }
        }
    }

    /// Cancels a request and notifies its caller.
    pub fn cancel(&mut self, id: SequenceId) {
        if self.scheduler.cancel(id) {
            if let Some(req) = self.active.remove(&id) {
                let _ = req.events.try_send(StreamEvent::Done {
                    reason: FinishReason::Cancelled,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: 0,
                });
            }
        }
    }

    /// Whether there is anything to do.
    pub fn has_work(&self) -> bool {
        self.scheduler.has_work()
    }

    /// Runs one engine step: schedule, forward, sample, emit.
    ///
    /// Returns the number of sequences that ran. Zero means the step was empty,
    /// which the driver treats as a cue to wait rather than spin.
    pub fn step(&mut self) -> Result<usize, EngineError> {
        // Retire anything that timed out before doing new work.
        for id in self.scheduler.expire_timeouts() {
            self.finish_request(id, FinishReason::Timeout);
        }

        let output = self.scheduler.schedule();
        if output.is_empty() {
            // Reap even on an empty step: a sequence may have been cancelled or
            // timed out into a finished state.
            self.reap();
            return Ok(0);
        }

        // Preempted sequences keep their caller channel; they will be
        // rescheduled, so nothing is emitted for them here.
        let batch = self.build_batch(&output)?;
        let forward = self.model.forward(&batch)?;

        let num_sequences = forward.num_sequences();
        for (row, &seq_id) in forward.sequence_ids.iter().enumerate() {
            // Prefill sequences that are not yet complete produce no token this
            // step; their logits are for a position mid-prompt.
            let still_prefilling = self
                .scheduler
                .running_sequence(seq_id)
                .is_some_and(|s| !s.is_prefill_complete());
            if still_prefilling {
                continue;
            }

            let Some(logits) = forward.row(row) else {
                return Err(EngineError::Internal(format!(
                    "forward output has no row {row} for {seq_id}"
                )));
            };

            self.sample_and_emit(seq_id, logits)?;
        }

        // Publish completed prefills so later requests can share their blocks.
        for sched in &output.scheduled {
            if sched.is_prefill {
                self.scheduler.commit_prefill(sched.id);
            }
        }

        self.reap();
        Ok(num_sequences)
    }

    /// Samples one token for a sequence and streams it to the caller.
    fn sample_and_emit(&mut self, seq_id: SequenceId, logits: &[f32]) -> Result<(), EngineError> {
        let Some(seq) = self.scheduler.running_sequence(seq_id) else {
            // Cancelled between scheduling and now. Nothing to do.
            return Ok(());
        };
        let context: Vec<TokenId> = seq.all_tokens().collect();
        let params = seq.params().clone();

        let Some(req) = self.active.get_mut(&seq_id) else {
            return Ok(());
        };

        // The sampler mutates logits in place, so it gets a scratch copy.
        let mut scratch = logits.to_vec();
        let token = req.sampler.sample(&mut scratch, &context, &params)?;
        let index = context.len().saturating_sub(req.prompt_len);

        let reason = self.scheduler.on_token(seq_id, token, &self.eos_ids);

        // The engine emits token ids; text conversion happens in the API layer,
        // which owns the per-request incremental decoder. Keeping the tokenizer
        // out of the step loop means detokenization never blocks the GPU.
        if let Some(req) = self.active.get(&seq_id) {
            let event = StreamEvent::Token {
                token,
                text: String::new(),
                index,
            };
            if req.events.try_send(event).is_err() {
                // The caller is gone or its buffer is full. A disconnected
                // client should not keep consuming GPU time, so cancel.
                tracing::debug!(sequence = %seq_id, "receiver gone, cancelling");
                self.cancel_silently(seq_id);
                return Ok(());
            }
        }

        if let Some(r) = reason {
            self.finish_request(seq_id, r);
        }
        Ok(())
    }

    /// Cancels without emitting, for a caller that has already disconnected.
    fn cancel_silently(&mut self, id: SequenceId) {
        self.scheduler.cancel(id);
        self.active.remove(&id);
    }

    /// Emits the terminal event for a sequence and drops its state.
    fn finish_request(&mut self, id: SequenceId, reason: FinishReason) {
        let completion_tokens = self
            .scheduler
            .running_sequence(id)
            .map(|s| s.output_len())
            .unwrap_or(0);

        if let Some(req) = self.active.remove(&id) {
            let _ = req.events.try_send(StreamEvent::Done {
                reason,
                prompt_tokens: req.prompt_len,
                completion_tokens,
            });
        }
    }

    /// Collects finished sequences, freeing their cache blocks and notifying
    /// any caller that has not already been told.
    fn reap(&mut self) {
        for seq in self.scheduler.reap_finished() {
            let id = seq.id();
            if let Some(req) = self.active.remove(&id) {
                let reason = seq.finish_reason().cloned().unwrap_or(FinishReason::Stop);
                let _ = req.events.try_send(StreamEvent::Done {
                    reason,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: seq.output_len(),
                });
            }
        }
    }

    /// Assembles the flat batch the model consumes.
    fn build_batch(
        &self,
        output: &orion_scheduler::SchedulerOutput,
    ) -> Result<ForwardBatch, EngineError> {
        let mut batch = ForwardBatch {
            tokens: Vec::new(),
            positions: Vec::new(),
            sequence_ids: Vec::new(),
            slot_token_counts: Vec::new(),
            context_lens: Vec::new(),
            block_tables: Vec::new(),
            has_prefill: false,
        };

        for sched in &output.scheduled {
            let Some(seq) = self.scheduler.running_sequence(sched.id) else {
                // Cancelled since scheduling; skip rather than fail the batch.
                continue;
            };
            let Some(table) = self.scheduler.cache().raw_block_table(sched.id) else {
                return Err(EngineError::Internal(format!(
                    "{} was scheduled without a block table",
                    sched.id
                )));
            };

            // For prefill, the tokens are the prompt slice this chunk covers.
            // For decode, it is the single most recent token.
            let tokens: Vec<TokenId> = if sched.is_prefill {
                let start = sched.start_position;
                let end = (start + sched.num_tokens).min(seq.prompt_len());
                seq.prompt()[start..end].to_vec()
            } else {
                match seq.output().last() {
                    Some(&t) => vec![t],
                    // A decode-scheduled sequence with no output yet takes its
                    // last prompt token, which happens right after prefill.
                    None => seq.prompt().last().copied().into_iter().collect(),
                }
            };

            if tokens.is_empty() {
                continue;
            }

            let count = tokens.len();
            batch
                .positions
                .extend((sched.start_position..sched.start_position + count).map(|p| p as u32));
            batch.tokens.extend(tokens);
            batch.sequence_ids.push(sched.id);
            batch.slot_token_counts.push(count);
            batch.context_lens.push(sched.start_position + count);
            batch.block_tables.push(table);
            batch.has_prefill |= sched.is_prefill;
        }

        batch.validate()?;
        Ok(batch)
    }
}

/// Spawns the engine on a dedicated thread and returns a handle to it.
///
/// `queue_depth` bounds the command channel. It is the outermost backpressure
/// valve: once full, `generate` blocks, and the API layer's own concurrency
/// limit turns that into a 429 rather than an unbounded backlog.
pub fn spawn(
    scheduler: Scheduler<KvCacheManager>,
    model: Arc<dyn LanguageModel>,
    queue_depth: usize,
) -> (EngineHandle, std::thread::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::channel::<Command>(queue_depth.max(1));

    let join = std::thread::Builder::new()
        .name("orion-engine".into())
        .spawn(move || {
            let mut engine = Engine::new(scheduler, model);
            tracing::info!("engine thread started");

            loop {
                // Drain every pending command without blocking, so a burst of
                // arrivals joins the same step rather than one per step.
                loop {
                    match rx.try_recv() {
                        Ok(Command::Generate(req)) => engine.submit(*req),
                        Ok(Command::Cancel(id)) => engine.cancel(id),
                        Ok(Command::Stats(reply)) => {
                            let _ = reply.send(engine.stats());
                        }
                        Ok(Command::Shutdown) => {
                            tracing::info!("engine shutting down");
                            return;
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            tracing::info!("engine handle dropped, stopping");
                            return;
                        }
                    }
                }

                if engine.has_work() {
                    if let Err(e) = engine.step() {
                        // A step failure is an engine-level fault. Log it and
                        // keep serving: one bad batch must not kill the server.
                        tracing::error!(error = %e, "engine step failed");
                    }
                } else {
                    // Idle: block for the next command rather than spinning.
                    match rx.blocking_recv() {
                        Some(Command::Generate(req)) => engine.submit(*req),
                        Some(Command::Cancel(id)) => engine.cancel(id),
                        Some(Command::Stats(reply)) => {
                            let _ = reply.send(engine.stats());
                        }
                        Some(Command::Shutdown) | None => return,
                    }
                }
            }
        })
        .expect("failed to spawn engine thread");

    (EngineHandle { tx }, join)
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_core::{Backend, DType, Device, ForwardOutput, ModelMetadata, SchedulerConfig};

    /// A model that returns fixed logits, so engine behaviour can be tested
    /// without any real weights.
    #[derive(Debug)]
    struct MockModel {
        meta: ModelMetadata,
        backend: MockBackend,
        /// Token the argmax always lands on.
        favoured: TokenId,
    }

    #[derive(Debug)]
    struct MockBackend;

    impl Backend for MockBackend {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn total_memory(&self) -> Option<u64> {
            None
        }
        fn available_memory(&self) -> Option<u64> {
            None
        }
        fn synchronize(&self) -> Result<(), EngineError> {
            Ok(())
        }
        fn supports_dtype(&self, _d: DType) -> bool {
            true
        }
    }

    impl MockModel {
        fn new(favoured: TokenId) -> Self {
            Self {
                meta: ModelMetadata {
                    architecture: "mock".into(),
                    name: "mock".into(),
                    hidden_size: 8,
                    num_layers: 1,
                    num_attention_heads: 1,
                    num_kv_heads: 1,
                    head_dim: 8,
                    vocab_size: 16,
                    max_position_embeddings: 512,
                    rope_theta: 10000.0,
                    rms_norm_eps: 1e-5,
                    dtype: DType::F32,
                    eos_token_ids: vec![15],
                    bos_token_id: Some(0),
                },
                backend: MockBackend,
                favoured,
            }
        }
    }

    impl LanguageModel for MockModel {
        fn metadata(&self) -> &ModelMetadata {
            &self.meta
        }
        fn backend(&self) -> &dyn Backend {
            &self.backend
        }
        fn forward(&self, batch: &ForwardBatch) -> Result<ForwardOutput, EngineError> {
            batch.validate()?;
            let n = batch.num_sequences();
            let v = self.meta.vocab_size;
            let mut logits = vec![0.0f32; n * v];
            for row in 0..n {
                logits[row * v + self.favoured as usize] = 10.0;
            }
            Ok(ForwardOutput {
                logits,
                vocab_size: v,
                sequence_ids: batch.sequence_ids.clone(),
            })
        }
    }

    fn engine(favoured: TokenId) -> Engine {
        let cache = KvCacheManager::new(64, 4, true);
        let scheduler = Scheduler::new(
            SchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: 64,
                max_model_len: Some(256),
                request_timeout_secs: None,
                ..Default::default()
            },
            cache,
        );
        Engine::new(scheduler, Arc::new(MockModel::new(favoured)))
    }

    fn greedy(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            max_tokens,
            ..Default::default()
        }
    }

    /// Submits a request and returns its id plus the receiving end.
    fn submit(
        e: &mut Engine,
        prompt: Vec<TokenId>,
        params: SamplingParams,
    ) -> (Result<SequenceId, EngineError>, mpsc::Receiver<StreamEvent>) {
        let (tx, rx) = mpsc::channel(256);
        let (acc_tx, mut acc_rx) = oneshot::channel();
        e.submit(GenerationRequest {
            prompt,
            params,
            events: tx,
            accepted: acc_tx,
        });
        let result = acc_rx
            .try_recv()
            .expect("admission is answered synchronously");
        (result, rx)
    }

    fn drain(rx: &mut mpsc::Receiver<StreamEvent>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            out.push(ev);
        }
        out
    }

    #[test]
    fn a_request_generates_until_max_tokens() {
        let mut e = engine(5);
        let (id, mut rx) = submit(&mut e, vec![1, 2, 3], greedy(4));
        assert!(id.is_ok());

        for _ in 0..20 {
            if !e.has_work() {
                break;
            }
            e.step().unwrap();
        }

        let events = drain(&mut rx);
        let tokens: Vec<_> = events
            .iter()
            .filter_map(|ev| match ev {
                StreamEvent::Token { token, .. } => Some(*token),
                _ => None,
            })
            .collect();

        assert_eq!(tokens.len(), 4, "should stop at max_tokens");
        assert!(tokens.iter().all(|&t| t == 5), "greedy should pick token 5");

        match events.last() {
            Some(StreamEvent::Done {
                reason,
                prompt_tokens,
                completion_tokens,
            }) => {
                assert_eq!(*reason, FinishReason::Length);
                assert_eq!(*prompt_tokens, 3);
                assert_eq!(*completion_tokens, 4);
            }
            other => panic!("expected a terminal Done, got {other:?}"),
        }
    }

    #[test]
    fn an_eos_token_stops_generation_early() {
        // The mock always favours 15, which is this model's EOS.
        let mut e = engine(15);
        let (_, mut rx) = submit(&mut e, vec![1, 2], greedy(100));

        for _ in 0..10 {
            if !e.has_work() {
                break;
            }
            e.step().unwrap();
        }

        let events = drain(&mut rx);
        let done = events
            .iter()
            .find(|e| matches!(e, StreamEvent::Done { .. }));
        match done {
            Some(StreamEvent::Done {
                reason,
                completion_tokens,
                ..
            }) => {
                assert_eq!(*reason, FinishReason::Stop);
                assert_eq!(*completion_tokens, 1, "should stop on the first EOS");
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn several_requests_are_served_concurrently() {
        let mut e = engine(5);
        let mut receivers = Vec::new();
        for _ in 0..4 {
            let (id, rx) = submit(&mut e, vec![1, 2], greedy(3));
            assert!(id.is_ok());
            receivers.push(rx);
        }

        let mut steps = 0;
        while e.has_work() && steps < 50 {
            e.step().unwrap();
            steps += 1;
        }

        for (i, mut rx) in receivers.into_iter().enumerate() {
            let events = drain(&mut rx);
            let n = events
                .iter()
                .filter(|ev| matches!(ev, StreamEvent::Token { .. }))
                .count();
            assert_eq!(n, 3, "request {i} generated {n} tokens");
            assert!(
                matches!(events.last(), Some(StreamEvent::Done { .. })),
                "request {i} did not terminate"
            );
        }
    }

    #[test]
    fn cache_blocks_are_released_when_requests_finish() {
        let mut e = engine(5);
        let free_before = e.stats().cache_free_blocks;

        let (_, _rx) = submit(&mut e, vec![1; 8], greedy(2));
        let mut steps = 0;
        while e.has_work() && steps < 30 {
            e.step().unwrap();
            steps += 1;
        }

        assert_eq!(
            e.stats().cache_free_blocks,
            free_before,
            "finished requests must return their blocks"
        );
        assert_eq!(e.active.len(), 0, "engine leaked per-request state");
    }

    #[test]
    fn invalid_sampling_parameters_are_rejected_at_submission() {
        let mut e = engine(5);
        let bad = SamplingParams {
            temperature: -1.0,
            ..Default::default()
        };
        let (result, _rx) = submit(&mut e, vec![1], bad);
        assert!(matches!(result, Err(EngineError::InvalidRequest(_))));
        assert_eq!(e.stats().waiting, 0, "a rejected request must not queue");
    }

    #[test]
    fn an_oversized_context_is_rejected_at_submission() {
        let mut e = engine(5);
        let (result, _rx) = submit(&mut e, vec![1; 300], greedy(10));
        assert!(matches!(
            result,
            Err(EngineError::ContextLengthExceeded { .. })
        ));
    }

    #[test]
    fn cancelling_frees_blocks_and_notifies_the_caller() {
        let mut e = engine(5);
        let (id, mut rx) = submit(&mut e, vec![1; 8], greedy(100));
        let id = id.unwrap();

        e.step().unwrap();
        assert!(e.stats().cache_free_blocks < 64);

        e.cancel(id);
        assert_eq!(e.stats().cache_free_blocks, 64, "cancel must free blocks");

        let events = drain(&mut rx);
        assert!(
            events.iter().any(|ev| matches!(
                ev,
                StreamEvent::Done {
                    reason: FinishReason::Cancelled,
                    ..
                }
            )),
            "caller should be told about the cancellation"
        );
    }

    #[test]
    fn cancelling_an_unknown_sequence_is_harmless() {
        let mut e = engine(5);
        e.cancel(SequenceId::from_raw(9999));
        assert_eq!(e.stats().running, 0);
    }

    #[test]
    fn an_empty_step_is_not_an_error() {
        let mut e = engine(5);
        assert!(!e.has_work());
        assert_eq!(e.step().unwrap(), 0);
    }

    #[test]
    fn a_long_prompt_is_prefilled_in_chunks_then_decodes() {
        let mut e = engine(5);
        // 100 prompt tokens against a 64-token budget forces chunking.
        let (id, mut rx) = submit(&mut e, vec![1; 100], greedy(2));
        assert!(id.is_ok());

        let mut steps = 0;
        while e.has_work() && steps < 50 {
            e.step().unwrap();
            steps += 1;
        }

        let events = drain(&mut rx);
        let tokens = events
            .iter()
            .filter(|ev| matches!(ev, StreamEvent::Token { .. }))
            .count();
        assert_eq!(tokens, 2, "chunked prefill must still produce max_tokens");
        assert!(steps > 2, "chunking should take several steps");
    }

    #[test]
    fn stats_reflect_engine_activity() {
        let mut e = engine(5);
        assert_eq!(e.stats().running, 0);

        // The receiver must be held: dropping it makes the engine treat the
        // caller as disconnected and cancel, which is correct behaviour but
        // not what this test is about.
        let (_id, _rx) = submit(&mut e, vec![1; 4], greedy(5));
        e.step().unwrap();

        let s = e.stats();
        assert_eq!(s.running, 1);
        assert!(s.cache_utilization() > 0.0);
        assert!(s.scheduler.total_admitted >= 1);
    }

    #[test]
    fn a_disconnected_caller_has_its_request_cancelled() {
        // A client that goes away must not keep consuming GPU time.
        let mut e = engine(5);
        let (tx, rx) = mpsc::channel(4);
        let (acc_tx, mut acc_rx) = oneshot::channel();
        e.submit(GenerationRequest {
            prompt: vec![1, 2],
            params: greedy(100),
            events: tx,
            accepted: acc_tx,
        });
        assert!(acc_rx.try_recv().unwrap().is_ok());

        drop(rx); // client disconnects

        let mut steps = 0;
        while e.has_work() && steps < 20 {
            e.step().unwrap();
            steps += 1;
        }
        assert!(!e.has_work(), "engine kept working for a gone client");
        assert_eq!(e.stats().cache_free_blocks, 64, "blocks must be released");
    }

    #[test]
    fn the_prefix_cache_serves_a_repeated_prompt() {
        let mut e = engine(5);
        let prompt: Vec<TokenId> = (0..16).collect();

        for _ in 0..2 {
            // Hold the receiver for the duration of the run; a dropped one
            // would cancel the request before it could commit its prefill.
            let (_id, _rx) = submit(&mut e, prompt.clone(), greedy(1));
            let mut steps = 0;
            while e.has_work() && steps < 20 {
                e.step().unwrap();
                steps += 1;
            }
        }

        let s = e.stats();
        assert!(
            s.prefix_cache_hits > 0,
            "an identical repeated prompt should hit the prefix cache"
        );
        assert!(s.prefix_cache_hit_rate() > 0.0);
    }
}
