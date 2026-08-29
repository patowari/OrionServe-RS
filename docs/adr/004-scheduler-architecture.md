# ADR 004: Continuous batching with decode-first ordering

**Status:** Accepted
**Date:** 2026-08-29

## Context

Decoding one token requires reading every weight in the model to do a handful
of matrix-vector products. The arithmetic intensity is so low that a GPU
serving one sequence is almost entirely idle, waiting on memory. Batching *n*
sequences reads those weights once and does *n* times the arithmetic, so
throughput scales close to linearly with batch size until something else binds.

Requests are heterogeneous: prompts span tens to tens of thousands of tokens,
and output length is unknown until generation stops. Any batching scheme that
assumes uniformity wastes most of what it batches.

## Decision

**Continuous batching.** Sequences join and leave the batch every step, rather
than a batch running to completion before the next forms.

**Decode-first ordering by default.** Running sequences get their token before
new prompts are admitted (`prioritize_decode`, default true).

**Chunked prefill.** Long prompts are split to fit the per-step token budget.

**Preemption by recompute**, evicting the newest sequence, returning it to the
*front* of the waiting queue.

**Two independent budgets:** `max_num_batched_tokens` (compute per step) and
`max_num_seqs` (resident sequences, each costing cache blocks).

## Alternatives considered

**Static batching.** Form a batch, run to completion, repeat. Rejected: the
batch runs until its longest member finishes, so with an order-of-magnitude
spread in output lengths most slots idle most of the time. It also blocks
arrivals for the duration of the current batch.

**Prefill-first ordering.** Better TTFT for new arrivals. Rejected as the
default because it produces visible stutter for clients already streaming, and
inter-token stalls are far more noticeable than a slightly higher TTFT that the
client has nothing to compare against. Available via configuration.

**Unchunked prefill.** Simpler, and avoids splitting attention across steps.
Rejected as the default because one 32k-token prompt would stall every
streaming client for the length of that forward pass. Still available, and
config validation rejects the combination where a max-length prompt could never
be scheduled.

**Preempting the oldest sequence.** Rejected: the oldest has generated the most
tokens, so evicting it discards the most completed work.

**Priority queues / SLO-aware ordering.** Rejected for now. FIFO is what
provides the starvation-freedom argument, and imposing a priority order breaks
it without a much more careful design. Left as `planned`.

## Consequences

**Good.** Batches stay full. A finished sequence frees its slot immediately.
Long prompts cannot monopolize the GPU. The whole policy is testable without a
GPU, because the scheduler is generic over `KvCacheManagerLike`.

**Bad.** Considerably more complex than static batching: sequences exist in
several states, and the interaction between chunked prefill and preemption is
subtle. Development surfaced exactly such a bug — a sequence part-way through
chunked prefill belonged to neither the decode path nor the admission path and
stalled permanently. It is fixed, documented in `docs/scheduler.md`, and pinned
by two regression tests.

**The load-bearing fairness property.** A preempted sequence goes to the front
of the waiting queue, not the back. Sending it to the back allows a steady
arrival stream to starve it indefinitely while repeatedly discarding its
prefill work — livelock presenting as a hung request. This is pinned by
`repeated_preemption_cannot_starve_a_sequence`.

**Deferred.** Swap-based preemption, priority classes, and speculative decoding
integration are `planned`. `PreemptionMode::Swap` is defined and validated but
not implemented.
