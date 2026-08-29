//! The sequence: the unit of work the scheduler and KV cache both operate on.
//!
//! A sequence owns its token history and its lifecycle state. It deliberately
//! does *not* own its KV blocks — the block table lives in the cache manager,
//! keyed by [`SequenceId`], so that cache ownership has exactly one home and
//! preemption can reclaim blocks without touching sequence state.

use std::time::Instant;

use crate::id::{SequenceId, TokenId};
use crate::sampling::SamplingParams;

/// Where a sequence is in its lifecycle.
///
/// The legal transitions are:
///
/// ```text
///   Waiting ──admit──> Prefilling ──> Decoding ──> Finished
///      ^                                 │
///      └──────────── preempt ────────────┘
/// ```
///
/// `Waiting -> Finished` is also legal (cancellation or timeout before the
/// sequence was ever scheduled). Everything else is a bug, enforced by
/// [`SequenceState::can_transition_to`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequenceState {
    /// Admitted to the engine, waiting for cache blocks and a batch slot.
    Waiting,
    /// Prompt tokens are being processed. With chunked prefill a sequence can
    /// stay here across several engine steps.
    Prefilling,
    /// Generating one token per engine step.
    Decoding,
    /// Terminal. See [`Sequence::finish_reason`] for why.
    Finished,
}

impl SequenceState {
    /// Whether `self -> next` is a legal transition.
    pub fn can_transition_to(self, next: SequenceState) -> bool {
        use SequenceState::*;
        matches!(
            (self, next),
            (Waiting, Prefilling)
                | (Waiting, Finished)
                | (Prefilling, Prefilling)
                | (Prefilling, Decoding)
                | (Prefilling, Waiting)
                | (Prefilling, Finished)
                | (Decoding, Decoding)
                | (Decoding, Waiting)
                | (Decoding, Finished)
        )
    }

    /// Whether the sequence currently occupies KV cache blocks and a batch slot.
    pub fn is_active(self) -> bool {
        matches!(self, SequenceState::Prefilling | SequenceState::Decoding)
    }

    /// Short label for metrics and log fields.
    pub fn as_str(self) -> &'static str {
        match self {
            SequenceState::Waiting => "waiting",
            SequenceState::Prefilling => "prefilling",
            SequenceState::Decoding => "decoding",
            SequenceState::Finished => "finished",
        }
    }
}

/// Why a sequence stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS or a caller-supplied stop token.
    Stop,
    /// Matched one of the caller's stop strings.
    StopSequence(String),
    /// Reached `max_tokens`.
    Length,
    /// Client disconnected or explicitly aborted.
    Cancelled,
    /// Deadline expired.
    Timeout,
    /// Engine-side failure; the message is for logs, not for the client.
    Error(String),
}

impl FinishReason {
    /// The `finish_reason` string in an OpenAI-compatible response.
    ///
    /// Note the deliberate collapse: internal distinctions like
    /// `StopSequence` map onto the small vocabulary clients expect.
    pub fn as_api_str(&self) -> &'static str {
        match self {
            FinishReason::Stop | FinishReason::StopSequence(_) => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
            FinishReason::Timeout => "timeout",
            FinishReason::Error(_) => "error",
        }
    }
}

/// Timing points captured per sequence, used to compute TTFT and TPOT without
/// a separate bookkeeping structure.
#[derive(Debug, Clone)]
pub struct SequenceTimings {
    /// When the request was accepted by the API layer.
    pub arrived_at: Instant,
    /// When the sequence first entered `Prefilling`.
    pub first_scheduled_at: Option<Instant>,
    /// When the first output token was produced.
    pub first_token_at: Option<Instant>,
    /// When the sequence reached `Finished`.
    pub finished_at: Option<Instant>,
}

impl SequenceTimings {
    fn new(arrived_at: Instant) -> Self {
        Self {
            arrived_at,
            first_scheduled_at: None,
            first_token_at: None,
            finished_at: None,
        }
    }

    /// Time to first token: arrival until the first generated token.
    ///
    /// This is measured from arrival, not from scheduling, because queueing
    /// delay is latency the client actually experiences.
    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        self.first_token_at.map(|t| t - self.arrived_at)
    }

    /// Time spent waiting in the queue before first being scheduled.
    pub fn queue_time(&self) -> Option<std::time::Duration> {
        self.first_scheduled_at.map(|t| t - self.arrived_at)
    }

    /// Mean time per output token, excluding the first.
    ///
    /// Returns `None` when fewer than two tokens were produced, since TPOT is
    /// undefined there rather than zero.
    pub fn time_per_output_token(&self, output_tokens: usize) -> Option<std::time::Duration> {
        let first = self.first_token_at?;
        let last = self.finished_at?;
        if output_tokens < 2 {
            return None;
        }
        Some((last - first) / (output_tokens as u32 - 1))
    }
}

/// One decoding sequence: prompt, generated tokens, state, and timings.
#[derive(Debug, Clone)]
pub struct Sequence {
    id: SequenceId,
    /// Prompt tokens. Never mutated after construction.
    prompt: Vec<TokenId>,
    /// Tokens produced by the model, in order.
    output: Vec<TokenId>,
    /// How many tokens of `prompt` already have KV entries computed. Advances
    /// in chunks under chunked prefill; equals `prompt.len()` once prefill is
    /// complete.
    computed_prefix_len: usize,
    state: SequenceState,
    finish_reason: Option<FinishReason>,
    params: SamplingParams,
    timings: SequenceTimings,
}

impl Sequence {
    /// Creates a sequence in the `Waiting` state.
    pub fn new(prompt: Vec<TokenId>, params: SamplingParams) -> Self {
        Self {
            id: SequenceId::next(),
            prompt,
            output: Vec::new(),
            computed_prefix_len: 0,
            state: SequenceState::Waiting,
            finish_reason: None,
            params,
            timings: SequenceTimings::new(Instant::now()),
        }
    }

    pub fn id(&self) -> SequenceId {
        self.id
    }

    pub fn state(&self) -> SequenceState {
        self.state
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    pub fn timings(&self) -> &SequenceTimings {
        &self.timings
    }

    pub fn prompt(&self) -> &[TokenId] {
        &self.prompt
    }

    pub fn output(&self) -> &[TokenId] {
        &self.output
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    /// Total logical length: everything that needs a KV entry.
    pub fn total_len(&self) -> usize {
        self.prompt.len() + self.output.len()
    }

    /// How many prompt tokens still need prefilling.
    pub fn remaining_prefill(&self) -> usize {
        self.prompt.len().saturating_sub(self.computed_prefix_len)
    }

    /// Whether every prompt token has been through the model.
    pub fn is_prefill_complete(&self) -> bool {
        self.computed_prefix_len >= self.prompt.len()
    }

    pub fn computed_prefix_len(&self) -> usize {
        self.computed_prefix_len
    }

    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finish_reason.as_ref()
    }

    pub fn is_finished(&self) -> bool {
        self.state == SequenceState::Finished
    }

    /// Attempts a state transition, rejecting illegal ones.
    ///
    /// Returns `false` and leaves the sequence untouched if the transition is
    /// not allowed. Callers in the scheduler treat `false` as a bug and log it
    /// rather than silently continuing.
    #[must_use]
    pub fn transition_to(&mut self, next: SequenceState) -> bool {
        if !self.state.can_transition_to(next) {
            return false;
        }
        if next == SequenceState::Prefilling && self.timings.first_scheduled_at.is_none() {
            self.timings.first_scheduled_at = Some(Instant::now());
        }
        self.state = next;
        true
    }

    /// Records that `n` more prompt tokens have been prefilled.
    ///
    /// Saturates at the prompt length so a mis-sized final chunk cannot push
    /// the prefix past the prompt.
    pub fn advance_prefill(&mut self, n: usize) {
        self.computed_prefix_len = (self.computed_prefix_len + n).min(self.prompt.len());
    }

    /// Appends a generated token and stamps the first-token timing.
    pub fn push_token(&mut self, token: TokenId) {
        if self.output.is_empty() {
            self.timings.first_token_at = Some(Instant::now());
        }
        self.output.push(token);
    }

    /// Moves the sequence to `Finished` with the given reason.
    ///
    /// Idempotent: the first reason wins, so a cancellation racing with a
    /// natural stop cannot rewrite history.
    pub fn finish(&mut self, reason: FinishReason) {
        if self.state == SequenceState::Finished {
            return;
        }
        self.state = SequenceState::Finished;
        self.finish_reason = Some(reason);
        self.timings.finished_at = Some(Instant::now());
    }

    /// Returns the sequence to `Waiting` after preemption, discarding prefill
    /// progress.
    ///
    /// Discarding `computed_prefix_len` is correct because preemption releases
    /// the sequence's KV blocks: the recomputed prefill must cover the whole
    /// prompt again. Generated tokens are *kept* and become part of the prompt
    /// on restart, so no output is lost.
    #[must_use]
    pub fn preempt(&mut self) -> bool {
        if !self.state.can_transition_to(SequenceState::Waiting) {
            return false;
        }
        self.state = SequenceState::Waiting;
        self.computed_prefix_len = 0;
        true
    }

    /// The token window the sampler applies repetition penalty over: the whole
    /// context, prompt included.
    pub fn all_tokens(&self) -> impl Iterator<Item = TokenId> + '_ {
        self.prompt
            .iter()
            .copied()
            .chain(self.output.iter().copied())
    }

    /// Whether the sequence has produced its caller-requested maximum.
    pub fn reached_max_tokens(&self) -> bool {
        self.output.len() >= self.params.max_tokens
    }

    /// Whether EOS is currently allowed to terminate the sequence, honouring
    /// `min_tokens`.
    pub fn may_stop_on_eos(&self) -> bool {
        self.output.len() >= self.params.min_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(prompt_len: usize) -> Sequence {
        Sequence::new(vec![1; prompt_len], SamplingParams::default())
    }

    #[test]
    fn new_sequence_starts_waiting_with_no_progress() {
        let s = seq(10);
        assert_eq!(s.state(), SequenceState::Waiting);
        assert_eq!(s.remaining_prefill(), 10);
        assert!(!s.is_prefill_complete());
        assert_eq!(s.total_len(), 10);
        assert!(s.finish_reason().is_none());
    }

    #[test]
    fn legal_lifecycle_transitions_are_accepted() {
        let mut s = seq(4);
        assert!(s.transition_to(SequenceState::Prefilling));
        assert!(s.transition_to(SequenceState::Decoding));
        s.finish(FinishReason::Length);
        assert!(s.is_finished());
    }

    #[test]
    fn illegal_transitions_are_rejected_without_mutating_state() {
        let mut s = seq(4);
        // Waiting cannot jump straight to Decoding.
        assert!(!s.transition_to(SequenceState::Decoding));
        assert_eq!(s.state(), SequenceState::Waiting);

        s.finish(FinishReason::Stop);
        // Finished is terminal.
        assert!(!s.transition_to(SequenceState::Decoding));
        assert!(!s.transition_to(SequenceState::Waiting));
        assert_eq!(s.state(), SequenceState::Finished);
    }

    #[test]
    fn chunked_prefill_advances_and_saturates() {
        let mut s = seq(10);
        assert!(s.transition_to(SequenceState::Prefilling));
        s.advance_prefill(4);
        assert_eq!(s.computed_prefix_len(), 4);
        assert_eq!(s.remaining_prefill(), 6);
        // An oversized final chunk must not overshoot.
        s.advance_prefill(100);
        assert_eq!(s.computed_prefix_len(), 10);
        assert_eq!(s.remaining_prefill(), 0);
        assert!(s.is_prefill_complete());
    }

    #[test]
    fn preemption_resets_prefill_but_keeps_generated_tokens() {
        let mut s = seq(8);
        assert!(s.transition_to(SequenceState::Prefilling));
        s.advance_prefill(8);
        assert!(s.transition_to(SequenceState::Decoding));
        s.push_token(42);
        s.push_token(43);

        assert!(s.preempt());
        assert_eq!(s.state(), SequenceState::Waiting);
        assert_eq!(s.computed_prefix_len(), 0, "prefill must be recomputed");
        assert_eq!(s.output(), &[42, 43], "generated tokens must survive");
    }

    #[test]
    fn finish_is_idempotent_and_keeps_the_first_reason() {
        let mut s = seq(2);
        s.finish(FinishReason::Stop);
        s.finish(FinishReason::Timeout);
        assert_eq!(s.finish_reason(), Some(&FinishReason::Stop));
    }

    #[test]
    fn first_token_timing_is_stamped_once() {
        let mut s = seq(2);
        assert!(s.timings().time_to_first_token().is_none());
        s.push_token(1);
        let first = s.timings().first_token_at;
        assert!(first.is_some());
        s.push_token(2);
        assert_eq!(s.timings().first_token_at, first);
    }

    #[test]
    fn tpot_is_undefined_below_two_tokens() {
        let mut s = seq(2);
        assert!(s.transition_to(SequenceState::Prefilling));
        s.push_token(1);
        s.finish(FinishReason::Length);
        assert!(s.timings().time_per_output_token(1).is_none());
        assert!(s.timings().time_per_output_token(2).is_some());
    }

    #[test]
    fn min_tokens_gates_eos_termination() {
        let params = SamplingParams {
            min_tokens: 2,
            ..Default::default()
        };
        let mut s = Sequence::new(vec![1, 2], params);
        assert!(!s.may_stop_on_eos());
        s.push_token(10);
        assert!(!s.may_stop_on_eos());
        s.push_token(11);
        assert!(s.may_stop_on_eos());
    }

    #[test]
    fn all_tokens_yields_prompt_then_output() {
        let mut s = Sequence::new(vec![1, 2], SamplingParams::default());
        s.push_token(3);
        assert_eq!(s.all_tokens().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn active_states_are_exactly_prefilling_and_decoding() {
        assert!(!SequenceState::Waiting.is_active());
        assert!(SequenceState::Prefilling.is_active());
        assert!(SequenceState::Decoding.is_active());
        assert!(!SequenceState::Finished.is_active());
    }
}
