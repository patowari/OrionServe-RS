# ADR 003: Paged block-based KV cache

**Status:** Accepted
**Date:** 2026-08-29

## Context

KV cache dominates GPU memory during inference. For Llama-3-8B in FP16 the
cache costs 128 KiB per token, so a single 8192-token sequence needs 1 GiB.
How that memory is managed determines how many concurrent requests fit, which
is the primary determinant of serving cost.

The difficulty is that a sequence's final length is unknown when it is
admitted. Generation stops on an EOS token the model has not produced yet.

## Decision

Divide the cache into fixed-size blocks (default 16 tokens, power of two,
range 4–256) allocated from a pool. Each sequence holds a block table mapping
logical positions to physical blocks, which need not be contiguous.

Blocks are reference-counted so sequences can share identical prompt prefixes.
Freed blocks enter a FIFO free list and retain their contents until reallocated,
which makes the free list double as an LRU eviction order for prefix caching.

## Alternatives considered

**Contiguous per-sequence buffers sized to max length.** The straightforward
approach. Rejected on measurement grounds: output length is unknown, so the
reservation must assume the worst case, and a request generating 100 tokens
against a 2048-token reservation wastes 95% of its allocation. It also suffers
external fragmentation — enough free memory in total, no single hole big enough.

**Contiguous buffers with growth by reallocation.** Avoids over-reservation but
requires copying the entire KV state on every growth, and growth happens
constantly during decode. Copying gigabytes of device memory per sequence per
growth event is not viable.

**Variable-size blocks / slab allocator with size classes.** Reduces internal
fragmentation further than fixed blocks. Rejected because it reintroduces
external fragmentation between size classes, complicates the attention kernel's
gather (which must now handle varying strides), and makes prefix sharing much
harder — two sequences can only share a block if their block *boundaries* align,
which variable sizing does not guarantee.

**Block size 8, 32, 64, 128.** See the table in `docs/kv-cache.md`. Below 4 the
block table rivals the data it indexes and gather coalescing degrades; above
256 internal waste becomes significant (at 256 sequences and block size 128,
up to 4 GiB wasted for the model above) and prefix sharing becomes too coarse
to fire on realistic shared prompts.

## Consequences

**Good.** No external fragmentation — every free block satisfies any request.
Internal fragmentation is bounded at `block_size - 1` tokens per sequence
regardless of final length. Prefix sharing becomes possible, and eviction
ordering comes free from the free list.

**Bad.** Attention kernels must gather through a block table rather than
reading contiguous memory, which costs some memory-access efficiency and makes
the kernels meaningfully more complex to write. The block table is per-sequence
metadata that must stay consistent with actual allocations — a class of bug
that does not exist with contiguous buffers.

**Mitigations applied.** Allocation is failure-atomic: a partial allocation
releases everything it took, so exhaustion cannot leak blocks. Newtype
`BlockId` and `SequenceId` prevent the two `usize` index spaces being confused.
Double-release returns an error rather than silently underflowing a refcount.

**Deferred.** Copy-on-write for forked sequences, a CPU swap tier, and sizing
the pool from actual device memory all remain `planned`.
