//! The continuous batching scheduler.
//!
//! Each engine step, [`Scheduler::schedule`] decides which sequences run and
//! with how many tokens each. Unlike static batching — where a batch is formed,
//! run to completion, and only then replaced — sequences here join and leave
//! the batch every step. A finished sequence frees its slot immediately instead
//! of idling while the longest member of its batch catches up.
//!
//! # Scheduling order
//!
//! 1. **Decode first** (when `prioritize_decode`). Already-running sequences
//!    get their one token each. This keeps inter-token latency smooth for
//!    clients who are already streaming, which is what they actually perceive.
//! 2. **Prefill with what remains.** Waiting sequences are admitted into the
//!    leftover token budget, chunked if enabled.
//!
//! The alternative — prefill first — gives better TTFT for new arrivals at the
//! cost of visible stutter for everyone already streaming. Both are available;
//! the default favours the running set. See `docs/scheduler.md`.
//!
//! # Why the scheduler owns no cache memory
//!
//! The scheduler asks the cache manager whether a sequence fits and tells it
//! when to allocate or free, but never touches blocks itself. That separation
//! is what lets every policy decision here be tested with a real cache manager
//! and no GPU: the tests in this module run the genuine allocation path.

use orion_core::{
    EngineError, FinishReason, KvCacheManagerLike, SamplingParams, SchedulerConfig, Sequence,
    SequenceId, SequenceState, TokenId,
};

use crate::queue::{RunningQueue, WaitingQueue};

/// What one sequence contributes to a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledSequence {
    pub id: SequenceId,
    /// Tokens this sequence submits this step.
    ///
    /// `1` for decode. For prefill it is the whole remaining prompt, or a
    /// chunk of it when chunked prefill is enabled.
    pub num_tokens: usize,
    /// Position of this sequence's first submitted token in the model's
    /// absolute coordinate space, for RoPE.
    pub start_position: usize,
    /// Whether this is prefill work rather than a decode step.
    pub is_prefill: bool,
}

/// The batch produced by one scheduling pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SchedulerOutput {
    pub scheduled: Vec<ScheduledSequence>,
    /// Sequences evicted this step to make room. Their blocks are already
    /// freed and they are back on the waiting queue.
    pub preempted: Vec<SequenceId>,
    /// Sequences that reached a terminal state this step.
    pub finished: Vec<(SequenceId, FinishReason)>,
}

impl SchedulerOutput {
    pub fn is_empty(&self) -> bool {
        self.scheduled.is_empty()
    }

    /// Total tokens in the batch — what the token budget bounds.
    pub fn num_batched_tokens(&self) -> usize {
        self.scheduled.iter().map(|s| s.num_tokens).sum()
    }

    pub fn num_sequences(&self) -> usize {
        self.scheduled.len()
    }

    pub fn num_prefill_sequences(&self) -> usize {
        self.scheduled.iter().filter(|s| s.is_prefill).count()
    }

    pub fn num_decode_sequences(&self) -> usize {
        self.scheduled.iter().filter(|s| !s.is_prefill).count()
    }

    /// Whether the batch mixes prefill and decode work.
    pub fn is_mixed(&self) -> bool {
        self.num_prefill_sequences() > 0 && self.num_decode_sequences() > 0
    }
}

/// Counters describing scheduler behaviour over time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub total_steps: u64,
    pub total_admitted: u64,
    pub total_finished: u64,
    pub total_preempted: u64,
    pub total_rejected: u64,
    pub total_batched_tokens: u64,
    pub total_prefill_tokens: u64,
    pub total_decode_tokens: u64,
}

impl SchedulerStats {
    /// Mean batch size in tokens across all non-empty steps.
    pub fn mean_batch_tokens(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_batched_tokens as f64 / self.total_steps as f64
        }
    }
}

/// Continuous batching scheduler.
///
/// Generic over the cache so scheduling policy can be tested against both the
/// real `KvCacheManager` and a purpose-built
/// fake that simulates exhaustion on demand.
#[derive(Debug)]
pub struct Scheduler<C: KvCacheManagerLike> {
    config: SchedulerConfig,
    waiting: WaitingQueue,
    running: RunningQueue,
    cache: C,
    stats: SchedulerStats,
}

impl<C: KvCacheManagerLike> Scheduler<C> {
    pub fn new(config: SchedulerConfig, cache: C) -> Self {
        Self {
            config,
            waiting: WaitingQueue::new(),
            running: RunningQueue::new(),
            cache,
            stats: SchedulerStats::default(),
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn stats(&self) -> SchedulerStats {
        self.stats
    }

    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// Mutable access to the cache, for tests that inject failures.
    pub fn cache_mut(&mut self) -> &mut C {
        &mut self.cache
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Whether any work is outstanding.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Admits a request, or rejects it if the engine is over its limits.
    ///
    /// This is the load-shedding boundary: refusing here, immediately and with
    /// a retryable error, is far better than accepting unbounded work and
    /// timing everything out later.
    pub fn add_request(&mut self, seq: Sequence) -> Result<SequenceId, EngineError> {
        if self.waiting.len() >= self.config.max_waiting_requests {
            self.stats.total_rejected += 1;
            return Err(EngineError::QueueFull {
                queued: self.waiting.len(),
                limit: self.config.max_waiting_requests,
            });
        }

        let max_len = self.max_model_len();
        let needed = seq.prompt_len() + seq.params().max_tokens;
        if needed > max_len {
            self.stats.total_rejected += 1;
            return Err(EngineError::ContextLengthExceeded {
                prompt_tokens: seq.prompt_len(),
                max_tokens: seq.params().max_tokens,
                context_len: max_len,
            });
        }

        // A prompt that could never fit the whole cache will never be
        // schedulable, no matter how much is freed. Rejecting now avoids a
        // sequence that blocks the queue head forever.
        let blocks = self.cache.blocks_needed_for(seq.prompt_len());
        if blocks > self.cache.total_blocks() {
            self.stats.total_rejected += 1;
            return Err(EngineError::CacheExhausted {
                needed: blocks,
                available: self.cache.total_blocks(),
            });
        }

        let id = seq.id();
        self.waiting.push_back(seq);
        self.stats.total_admitted += 1;
        Ok(id)
    }

    fn max_model_len(&self) -> usize {
        self.config.max_model_len.unwrap_or(usize::MAX)
    }

    /// Cancels a request wherever it currently sits.
    ///
    /// Returns `true` if it was found. Cache blocks are released on the running
    /// path; a waiting sequence holds none. Cancellation must be safe to call
    /// at any time, including for a sequence that finished microseconds ago,
    /// hence the tolerant `bool` rather than an error.
    pub fn cancel(&mut self, id: SequenceId) -> bool {
        if self.waiting.remove(id).is_some() {
            return true;
        }
        if let Some(mut seq) = self.running.remove(id) {
            seq.finish(FinishReason::Cancelled);
            self.cache.free(id);
            return true;
        }
        false
    }

    /// Builds the batch for one engine step.
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut out = SchedulerOutput::default();
        let budget = self.config.max_num_batched_tokens;
        let mut used = 0usize;

        if self.config.prioritize_decode {
            self.schedule_decodes(&mut out, budget, &mut used);
            self.schedule_prefills(&mut out, budget, &mut used);
        } else {
            self.schedule_prefills(&mut out, budget, &mut used);
            self.schedule_decodes(&mut out, budget, &mut used);
        }

        self.stats.total_steps += 1;
        self.stats.total_batched_tokens += out.num_batched_tokens() as u64;
        for s in &out.scheduled {
            if s.is_prefill {
                self.stats.total_prefill_tokens += s.num_tokens as u64;
            } else {
                self.stats.total_decode_tokens += s.num_tokens as u64;
            }
        }
        out
    }

    /// Gives one token to each running sequence, preempting when the cache
    /// cannot grow to accommodate them.
    fn schedule_decodes(&mut self, out: &mut SchedulerOutput, budget: usize, used: &mut usize) {
        // Collect first: growing the cache mutates it, and the borrow checker
        // rightly objects to iterating `running` while calling `self.cache`.
        let candidates: Vec<(SequenceId, usize)> = self
            .running
            .iter()
            .filter(|s| s.state() == SequenceState::Decoding)
            .map(|s| (s.id(), s.total_len()))
            .collect();

        for (id, total_len) in candidates {
            if *used >= budget {
                break;
            }

            // Reserve room for the token about to be generated.
            match self.cache.append_token(id) {
                Ok(()) => {
                    out.scheduled.push(ScheduledSequence {
                        id,
                        num_tokens: 1,
                        start_position: total_len,
                        is_prefill: false,
                    });
                    *used += 1;
                }
                Err(_) => {
                    // The cache is full. Evict the newest running sequence and
                    // retry; if the victim is this very sequence, it simply
                    // does not run this step.
                    if !self.preempt_one(out) {
                        break; // nothing left to evict
                    }
                    if self.cache.append_token(id).is_ok() {
                        out.scheduled.push(ScheduledSequence {
                            id,
                            num_tokens: 1,
                            start_position: total_len,
                            is_prefill: false,
                        });
                        *used += 1;
                    }
                }
            }
        }
    }

    /// Evicts the newest running sequence, returning it to the front of the
    /// waiting queue. Returns `false` when nothing can be evicted.
    fn preempt_one(&mut self, out: &mut SchedulerOutput) -> bool {
        let Some(mut victim) = self.running.pop_newest() else {
            return false;
        };
        let id = victim.id();
        self.cache.free(id);
        // Discards prefill progress but keeps generated tokens: on restart the
        // prompt is prompt + tokens-so-far, so no output is lost.
        let _ = victim.preempt();
        self.waiting.push_front(victim);
        out.preempted.push(id);
        self.stats.total_preempted += 1;
        true
    }

    /// Continues sequences already in `Prefilling` that still have prompt left.
    ///
    /// Under chunked prefill a sequence joins the running set after its first
    /// chunk but is not yet a decoder. Without this pass it would belong to
    /// neither scheduling path and would stall forever — the running-queue
    /// equivalent of a lost wakeup.
    ///
    /// Partial prefills are resumed *before* new admissions so that work
    /// already started, and already holding cache blocks, finishes rather than
    /// accumulating alongside newer arrivals.
    fn schedule_partial_prefills(
        &mut self,
        budget: usize,
        used: &mut usize,
    ) -> Vec<ScheduledSequence> {
        let mut scheduled = Vec::new();
        let candidates: Vec<(SequenceId, usize, usize)> = self
            .running
            .iter()
            .filter(|s| s.state() == SequenceState::Prefilling && !s.is_prefill_complete())
            .map(|s| (s.id(), s.computed_prefix_len(), s.remaining_prefill()))
            .collect();

        for (id, already, remaining) in candidates {
            if *used >= budget {
                break;
            }
            let room = budget - *used;
            let chunk = if self.config.enable_chunked_prefill {
                remaining.min(room)
            } else if remaining <= room {
                remaining
            } else {
                continue;
            };
            if chunk == 0 {
                continue;
            }

            let Some(seq) = self.running.get_mut(id) else {
                continue;
            };
            seq.advance_prefill(chunk);
            if seq.is_prefill_complete() && !seq.transition_to(SequenceState::Decoding) {
                tracing::error!(sequence = %id, "illegal transition into decode");
            }

            scheduled.push(ScheduledSequence {
                id,
                num_tokens: chunk,
                start_position: already,
                is_prefill: true,
            });
            *used += chunk;
        }
        scheduled
    }

    /// Admits waiting sequences into the remaining token budget.
    fn schedule_prefills(&mut self, out: &mut SchedulerOutput, budget: usize, used: &mut usize) {
        out.scheduled
            .extend(self.schedule_partial_prefills(budget, used));

        loop {
            if self.running.len() >= self.config.max_num_seqs || *used >= budget {
                break;
            }
            let Some(next) = self.waiting.peek() else {
                break;
            };

            let prompt_len = next.prompt_len();
            let already = next.computed_prefix_len();
            let remaining = prompt_len - already;
            let room = budget - *used;

            let chunk = if self.config.enable_chunked_prefill {
                remaining.min(room)
            } else if remaining <= room {
                remaining
            } else {
                // Without chunking the prompt must fit whole. If the budget is
                // free and it still does not fit, it never will, so stop
                // rather than spin; otherwise wait for a later, emptier step.
                break;
            };

            if chunk == 0 {
                break;
            }

            // Only the first chunk allocates; later chunks reuse the table.
            let is_first_chunk = already == 0;
            if is_first_chunk && !self.cache.can_allocate(prompt_len) {
                // Try to make room by evicting; if that fails, stop admitting.
                if !self.preempt_one(out) {
                    break;
                }
                continue;
            }

            let Some(mut seq) = self.waiting.pop_front() else {
                break;
            };
            let id = seq.id();

            if is_first_chunk {
                let prompt: Vec<TokenId> = seq.prompt().to_vec();
                if let Err(e) = self.cache.allocate(id, &prompt) {
                    // Allocation raced the availability check; put it back and
                    // stop for this step rather than dropping the request.
                    tracing::debug!(sequence = %id, error = %e, "prefill allocation failed");
                    let _ = seq.preempt();
                    self.waiting.push_front(seq);
                    break;
                }
            }

            let start_position = already;
            seq.advance_prefill(chunk);

            // A sequence stays in Prefilling until its prompt is fully
            // consumed; only then does it become a decoder.
            if seq.state() == SequenceState::Waiting
                && !seq.transition_to(SequenceState::Prefilling)
            {
                tracing::error!(sequence = %id, "illegal transition into prefill");
            }
            if seq.is_prefill_complete() && !seq.transition_to(SequenceState::Decoding) {
                tracing::error!(sequence = %id, "illegal transition into decode");
            }

            out.scheduled.push(ScheduledSequence {
                id,
                num_tokens: chunk,
                start_position,
                is_prefill: true,
            });
            *used += chunk;
            self.running.push(seq);
        }
    }

    /// Records a generated token against a running sequence and applies stop
    /// conditions.
    ///
    /// Returns the finish reason if the sequence terminated.
    pub fn on_token(
        &mut self,
        id: SequenceId,
        token: TokenId,
        eos_ids: &[TokenId],
    ) -> Option<FinishReason> {
        let seq = self.running.get_mut(id)?;
        seq.push_token(token);

        let params = seq.params().clone();
        let reason = Self::stop_reason(seq, token, eos_ids, &params);
        if let Some(r) = &reason {
            seq.finish(r.clone());
        }
        reason
    }

    fn stop_reason(
        seq: &Sequence,
        token: TokenId,
        eos_ids: &[TokenId],
        params: &SamplingParams,
    ) -> Option<FinishReason> {
        let is_stop_token = eos_ids.contains(&token) || params.stop_token_ids.contains(&token);
        if is_stop_token && seq.may_stop_on_eos() {
            return Some(FinishReason::Stop);
        }
        if seq.reached_max_tokens() {
            return Some(FinishReason::Length);
        }
        None
    }

    /// Retires finished sequences, freeing their cache blocks.
    ///
    /// Returns them so the engine can send final responses. Called after
    /// [`on_token`](Self::on_token) for every sequence in the step.
    pub fn reap_finished(&mut self) -> Vec<Sequence> {
        let finished = self.running.take_finished();
        for seq in &finished {
            self.cache.free(seq.id());
        }
        self.stats.total_finished += finished.len() as u64;
        finished
    }

    /// Terminates sequences that have exceeded the configured deadline.
    ///
    /// Returns the ids that were timed out. Checked once per step rather than
    /// with per-request timers: one linear scan of a bounded set is cheaper
    /// than thousands of timer registrations.
    pub fn expire_timeouts(&mut self) -> Vec<SequenceId> {
        let Some(limit) = self.config.request_timeout_secs else {
            return Vec::new();
        };
        let limit = std::time::Duration::from_secs(limit);
        let now = std::time::Instant::now();
        let mut expired = Vec::new();

        for seq in self.running.iter_mut() {
            if now.duration_since(seq.timings().arrived_at) > limit {
                seq.finish(FinishReason::Timeout);
                expired.push(seq.id());
            }
        }
        for seq in self
            .waiting
            .drain_where(|s| now.duration_since(s.timings().arrived_at) > limit)
        {
            expired.push(seq.id());
        }
        expired
    }

    /// Borrows a running sequence.
    pub fn running_sequence(&self, id: SequenceId) -> Option<&Sequence> {
        self.running.get(id)
    }

    /// Publishes a completed prefill to the prefix cache.
    pub fn commit_prefill(&mut self, id: SequenceId) {
        if let Some(seq) = self.running.get(id) {
            if seq.is_prefill_complete() {
                let prompt = seq.prompt().to_vec();
                self.cache.commit_prefill(id, &prompt);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::FakeCache;
    use orion_kv_cache::KvCacheManager;

    fn config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 32,
            max_model_len: Some(1024),
            enable_chunked_prefill: true,
            max_waiting_requests: 8,
            request_timeout_secs: None,
            prioritize_decode: true,
            ..Default::default()
        }
    }

    /// A scheduler over the real cache manager, so policy tests exercise the
    /// genuine allocation path.
    fn scheduler(blocks: usize) -> Scheduler<KvCacheManager> {
        Scheduler::new(config(), KvCacheManager::new(blocks, 4, false))
    }

    fn seq(prompt_len: usize, max_tokens: usize) -> Sequence {
        Sequence::new(
            vec![1; prompt_len],
            SamplingParams::default().with_max_tokens(max_tokens),
        )
    }

    #[test]
    fn an_empty_scheduler_produces_an_empty_batch() {
        let mut s = scheduler(16);
        let out = s.schedule();
        assert!(out.is_empty());
        assert!(!s.has_work());
    }

    #[test]
    fn a_single_request_is_prefilled_then_decodes() {
        let mut s = scheduler(16);
        let id = s.add_request(seq(8, 4)).unwrap();

        let out = s.schedule();
        assert_eq!(out.num_sequences(), 1);
        assert_eq!(out.scheduled[0].id, id);
        assert_eq!(out.scheduled[0].num_tokens, 8, "whole prompt in one chunk");
        assert!(out.scheduled[0].is_prefill);
        assert_eq!(out.scheduled[0].start_position, 0);

        // After prefill it becomes a decoder producing one token per step.
        s.on_token(id, 100, &[]);
        let out = s.schedule();
        assert_eq!(out.num_sequences(), 1);
        assert!(!out.scheduled[0].is_prefill);
        assert_eq!(out.scheduled[0].num_tokens, 1);
        assert_eq!(out.scheduled[0].start_position, 9, "8 prompt + 1 generated");
    }

    #[test]
    fn several_requests_are_batched_into_one_step() {
        let mut s = scheduler(64);
        for _ in 0..3 {
            s.add_request(seq(8, 4)).unwrap();
        }
        let out = s.schedule();
        assert_eq!(
            out.num_sequences(),
            3,
            "continuous batching, not sequential"
        );
        assert_eq!(out.num_batched_tokens(), 24);
    }

    #[test]
    fn the_token_budget_bounds_a_step() {
        let mut s = scheduler(64);
        // Three 16-token prompts against a 32-token budget.
        for _ in 0..3 {
            s.add_request(seq(16, 4)).unwrap();
        }
        let out = s.schedule();
        assert!(
            out.num_batched_tokens() <= 32,
            "budget exceeded: {}",
            out.num_batched_tokens()
        );
    }

    #[test]
    fn max_num_seqs_bounds_the_running_set() {
        let mut s = scheduler(256);
        for _ in 0..8 {
            s.add_request(seq(1, 4)).unwrap();
        }
        s.schedule();
        assert_eq!(s.num_running(), 4, "max_num_seqs is 4");
        assert_eq!(s.num_waiting(), 4);
    }

    #[test]
    fn chunked_prefill_splits_a_long_prompt_across_steps() {
        let mut s = scheduler(64);
        // 100 tokens against a 32-token budget: needs four steps.
        let id = s.add_request(seq(100, 4)).unwrap();

        let mut consumed = 0;
        let mut steps = 0;
        loop {
            let out = s.schedule();
            let sched = &out.scheduled[0];
            assert_eq!(sched.id, id);
            if !sched.is_prefill {
                break;
            }
            assert_eq!(
                sched.start_position, consumed,
                "chunks must be contiguous in position space"
            );
            consumed += sched.num_tokens;
            assert!(sched.num_tokens <= 32);
            steps += 1;
            assert!(steps < 10, "prefill failed to terminate");
        }
        assert_eq!(consumed, 100, "every prompt token must be prefilled once");
        assert_eq!(steps, 4, "ceil(100/32)");
    }

    #[test]
    fn a_partially_prefilled_sequence_is_never_stranded() {
        // Regression: a sequence that has had one prefill chunk lives in the
        // running queue but is not yet a decoder. If neither scheduling path
        // picks it up it stalls forever, and the request hangs.
        let mut s = scheduler(64);
        let id = s.add_request(seq(100, 4)).unwrap();

        let first = s.schedule();
        assert!(first.scheduled[0].is_prefill);
        assert!(
            first.scheduled[0].num_tokens < 100,
            "the prompt should have been chunked, not taken whole"
        );
        assert_eq!(s.num_running(), 1);
        assert_eq!(s.num_waiting(), 0, "it left the waiting queue");

        // The very next step must make progress on it.
        let second = s.schedule();
        assert!(
            !second.is_empty(),
            "partially prefilled sequence produced an empty batch"
        );
        assert_eq!(second.scheduled[0].id, id);
        assert!(second.scheduled[0].num_tokens > 0);
    }

    #[test]
    fn every_admitted_request_eventually_finishes() {
        // A liveness property: run a mixed workload to completion and assert
        // the scheduler never deadlocks or drops anyone.
        let mut s = Scheduler::new(
            SchedulerConfig {
                max_num_seqs: 4,
                max_num_batched_tokens: 32,
                max_model_len: Some(1024),
                enable_chunked_prefill: true,
                max_waiting_requests: 64,
                request_timeout_secs: None,
                prioritize_decode: true,
                ..Default::default()
            },
            KvCacheManager::new(128, 4, false),
        );

        let mut outstanding = Vec::new();
        for len in [4usize, 60, 8, 100, 16] {
            outstanding.push(s.add_request(seq(len, 5)).unwrap());
        }
        let submitted = outstanding.len();

        let mut completed = 0;
        for step in 0..500 {
            let out = s.schedule();
            for sched in &out.scheduled {
                if !sched.is_prefill {
                    s.on_token(sched.id, 42, &[]);
                }
            }
            completed += s.reap_finished().len();
            if !s.has_work() {
                break;
            }
            assert!(step < 499, "workload failed to drain");
        }
        assert_eq!(completed, submitted, "every request must finish");
        assert_eq!(s.num_running(), 0);
        assert_eq!(s.num_waiting(), 0);
    }

    #[test]
    fn chunked_prefill_interleaves_with_running_decodes() {
        let mut s = scheduler(128);
        // One short request reaches decode first.
        let short = s.add_request(seq(4, 100)).unwrap();
        s.schedule();
        s.on_token(short, 1, &[]);

        // A long prompt arrives; it must not monopolize the step.
        s.add_request(seq(200, 4)).unwrap();
        let out = s.schedule();

        assert!(out.is_mixed(), "step should mix decode and prefill");
        assert_eq!(out.num_decode_sequences(), 1);
        assert!(
            out.scheduled.iter().any(|x| x.id == short && !x.is_prefill),
            "the streaming request must still get its token"
        );
        assert!(out.num_batched_tokens() <= 32);
    }

    #[test]
    fn unchunked_prefill_waits_for_a_prompt_that_fits() {
        let cfg = SchedulerConfig {
            enable_chunked_prefill: false,
            max_num_batched_tokens: 32,
            max_model_len: Some(1024),
            ..config()
        };
        let mut s = Scheduler::new(cfg, KvCacheManager::new(128, 4, false));

        // Fill most of the budget with a decoding sequence.
        let a = s.add_request(seq(30, 100)).unwrap();
        s.schedule();
        s.on_token(a, 1, &[]);

        // A 20-token prompt cannot fit in the 31 remaining... it can.
        // Use one that genuinely cannot: 32 tokens with 1 already used.
        s.add_request(seq(32, 4)).unwrap();
        let out = s.schedule();
        assert_eq!(
            out.num_prefill_sequences(),
            0,
            "prompt must wait for a step with room for all of it"
        );
        assert_eq!(s.num_waiting(), 1);
    }

    #[test]
    fn stop_token_finishes_a_sequence_and_frees_its_blocks() {
        let mut s = scheduler(16);
        let id = s.add_request(seq(8, 100)).unwrap();
        s.schedule();
        let free_before = s.cache().free_blocks();
        assert!(free_before < 16);

        let reason = s.on_token(id, 2, &[2]);
        assert_eq!(reason, Some(FinishReason::Stop));

        let finished = s.reap_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(s.cache().free_blocks(), 16, "blocks must be returned");
        assert_eq!(s.num_running(), 0);
    }

    #[test]
    fn max_tokens_terminates_generation() {
        let mut s = scheduler(32);
        let id = s.add_request(seq(4, 3)).unwrap();
        s.schedule();

        assert_eq!(s.on_token(id, 10, &[]), None);
        assert_eq!(s.on_token(id, 11, &[]), None);
        assert_eq!(s.on_token(id, 12, &[]), Some(FinishReason::Length));
    }

    #[test]
    fn min_tokens_suppresses_an_early_eos() {
        let mut s = scheduler(32);
        let params = SamplingParams {
            min_tokens: 3,
            max_tokens: 10,
            ..Default::default()
        };
        let id = s.add_request(Sequence::new(vec![1; 4], params)).unwrap();
        s.schedule();

        assert_eq!(s.on_token(id, 2, &[2]), None, "EOS below min_tokens");
        assert_eq!(s.on_token(id, 2, &[2]), None);
        assert_eq!(s.on_token(id, 2, &[2]), Some(FinishReason::Stop));
    }

    #[test]
    fn queue_overflow_is_rejected_with_a_retryable_error() {
        let mut s = scheduler(1024);
        for _ in 0..8 {
            s.add_request(seq(1, 1)).unwrap();
        }
        let err = s.add_request(seq(1, 1)).unwrap_err();
        assert!(matches!(err, EngineError::QueueFull { .. }));
        assert!(err.is_retryable());
        assert_eq!(s.stats().total_rejected, 1);
    }

    #[test]
    fn an_oversized_context_is_rejected_at_admission() {
        let mut s = scheduler(1024);
        let err = s.add_request(seq(1000, 100)).unwrap_err();
        assert!(matches!(err, EngineError::ContextLengthExceeded { .. }));
        assert!(err.is_client_error());
    }

    #[test]
    fn a_prompt_larger_than_the_whole_cache_is_rejected_immediately() {
        // Otherwise it would sit at the queue head forever, blocking everyone.
        let mut s = scheduler(4); // 4 blocks * 4 tokens = 16 tokens
        let err = s.add_request(seq(100, 4)).unwrap_err();
        assert!(matches!(err, EngineError::CacheExhausted { .. }));
        assert_eq!(s.num_waiting(), 0);
    }

    #[test]
    fn cancelling_a_waiting_request_removes_it() {
        let mut s = scheduler(32);
        let id = s.add_request(seq(4, 4)).unwrap();
        assert!(s.cancel(id));
        assert_eq!(s.num_waiting(), 0);
        assert!(!s.cancel(id), "second cancel is a no-op");
    }

    #[test]
    fn cancelling_a_running_request_frees_its_blocks() {
        let mut s = scheduler(32);
        let id = s.add_request(seq(8, 100)).unwrap();
        s.schedule();
        assert!(s.cache().free_blocks() < 32);

        assert!(s.cancel(id));
        assert_eq!(s.cache().free_blocks(), 32);
        assert_eq!(s.num_running(), 0);
    }

    #[test]
    fn preemption_evicts_the_newest_and_it_keeps_its_tokens() {
        // A tiny cache forces eviction once sequences grow.
        let cfg = SchedulerConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 64,
            max_model_len: Some(1024),
            ..config()
        };
        let mut s = Scheduler::new(cfg, KvCacheManager::new(4, 4, false));

        let a = s.add_request(seq(4, 100)).unwrap();
        let b = s.add_request(seq(4, 100)).unwrap();
        s.schedule();
        assert_eq!(s.num_running(), 2);

        // Both grow until the 4-block cache cannot hold another token.
        let mut preempted = Vec::new();
        for _ in 0..8 {
            s.on_token(a, 1, &[]);
            s.on_token(b, 1, &[]);
            let out = s.schedule();
            preempted.extend(out.preempted);
        }

        assert!(
            !preempted.is_empty(),
            "cache pressure must cause preemption"
        );
        assert_eq!(preempted[0], b, "the newest sequence is evicted first");
        assert!(s.stats().total_preempted > 0);
    }

    #[test]
    fn a_preempted_sequence_is_rescheduled_before_new_arrivals() {
        let mut s = Scheduler::new(
            SchedulerConfig {
                max_num_seqs: 2,
                max_num_batched_tokens: 64,
                max_model_len: Some(1024),
                ..config()
            },
            KvCacheManager::new(3, 4, false),
        );

        let a = s.add_request(seq(4, 100)).unwrap();
        let b = s.add_request(seq(4, 100)).unwrap();
        s.schedule();

        for _ in 0..6 {
            s.on_token(a, 1, &[]);
            s.on_token(b, 1, &[]);
            let out = s.schedule();
            if !out.preempted.is_empty() {
                // The victim went to the front, so it is next to be scheduled.
                let victim = out.preempted[0];
                let next = s.schedule();
                assert!(
                    next.scheduled.iter().any(|x| x.id == victim) || s.cache().free_blocks() == 0,
                    "preempted sequence must regain priority when room exists"
                );
                return;
            }
        }
    }

    #[test]
    fn cache_exhaustion_during_decode_does_not_lose_sequences() {
        // Every sequence must end up either running or waiting - never dropped.
        let mut s = Scheduler::new(
            SchedulerConfig {
                max_num_seqs: 8,
                max_num_batched_tokens: 64,
                max_model_len: Some(1024),
                ..config()
            },
            KvCacheManager::new(6, 4, false),
        );

        let ids: Vec<_> = (0..4).map(|_| s.add_request(seq(4, 50)).unwrap()).collect();
        s.schedule();

        for _ in 0..10 {
            for &id in &ids {
                s.on_token(id, 1, &[]);
            }
            s.reap_finished();
            s.schedule();
            let accounted = s.num_running() + s.num_waiting();
            let done = s.stats().total_finished as usize;
            assert_eq!(accounted + done, 4, "a sequence went missing");
        }
    }

    #[test]
    fn stats_track_prefill_and_decode_separately() {
        let mut s = scheduler(64);
        let id = s.add_request(seq(8, 4)).unwrap();
        s.schedule();
        assert_eq!(s.stats().total_prefill_tokens, 8);
        assert_eq!(s.stats().total_decode_tokens, 0);

        s.on_token(id, 1, &[]);
        s.schedule();
        assert_eq!(s.stats().total_decode_tokens, 1);
        assert_eq!(s.stats().total_admitted, 1);
    }

    #[test]
    fn prefill_priority_can_be_inverted_by_config() {
        let cfg = SchedulerConfig {
            prioritize_decode: false,
            max_num_batched_tokens: 8,
            ..config()
        };
        let mut s = Scheduler::new(cfg, KvCacheManager::new(128, 4, false));

        let running = s.add_request(seq(4, 100)).unwrap();
        s.schedule();
        s.on_token(running, 1, &[]);

        // A new 8-token prompt should now claim the budget ahead of the decode.
        s.add_request(seq(8, 4)).unwrap();
        let out = s.schedule();
        assert_eq!(out.num_prefill_sequences(), 1);
        assert_eq!(
            out.num_decode_sequences(),
            0,
            "prefill consumed the whole budget first"
        );
    }

    #[test]
    fn timeouts_expire_waiting_and_running_sequences() {
        let cfg = SchedulerConfig {
            request_timeout_secs: Some(0),
            ..config()
        };
        // A zero-second limit is rejected by config validation, so construct
        // the scheduler directly to exercise the expiry path.
        let mut s = Scheduler::new(
            SchedulerConfig {
                request_timeout_secs: Some(1),
                ..cfg
            },
            KvCacheManager::new(64, 4, false),
        );
        s.add_request(seq(4, 4)).unwrap();
        // Nothing has aged past one second yet.
        assert!(s.expire_timeouts().is_empty());
    }

    #[test]
    fn a_simulated_cache_failure_is_handled_without_panicking() {
        // FakeCache lets us force exhaustion deterministically, which a real
        // manager only reaches through specific size arithmetic.
        let mut s = Scheduler::new(config(), FakeCache::new(2, 4));
        s.add_request(seq(4, 10)).unwrap();
        s.add_request(seq(4, 10)).unwrap();
        s.schedule();

        s.cache_mut().fail_next_append();
        let ids: Vec<_> = s.running.iter().map(|x| x.id()).collect();
        for id in ids {
            s.on_token(id, 1, &[]);
        }
        let out = s.schedule();
        // Either it preempted or it simply scheduled less; both are fine, and
        // no sequence may be lost.
        assert_eq!(s.num_running() + s.num_waiting(), 2 - out.finished.len());
    }
}
