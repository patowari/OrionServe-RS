# ADR 005: Backend trait with CPU implemented first

**Status:** Accepted
**Date:** 2026-08-29

## Context

The project targets GPU inference, but GPU work is the least verifiable part of
it: correctness bugs in a CUDA kernel are hard to localize, and there is no way
to tell whether a wrong answer came from the kernel, the cache indexing, or the
scheduler unless the non-GPU parts are already known good.

The development machine for this milestone has no NVIDIA GPU and no CUDA
toolkit, which makes the sequencing question concrete rather than theoretical.

## Decision

A narrow `Backend` trait describing the *device* — name, device id, memory
reporting, synchronization, dtype support — and **not** the operations.
Operations belong to the model implementation, which is generic over the
backend it runs on.

CPU is implemented first and remains the correctness reference. CUDA is added
only after model output is verified against it.

`Backend::synchronize` exists specifically so benchmarks are honest: without
it, an asynchronous launch queue makes kernels appear instantaneous.

## Alternatives considered

**A wide trait containing every operation** (`matmul`, `rmsnorm`, `rope`, …).
Rejected: every new operation would have to be implemented for every backend
before anything compiles, and the CPU backend would accumulate operations that
exist only to satisfy the trait.

**No abstraction — write directly against one tensor library.** Rejected
because it makes the eventual CUDA backend a rewrite rather than an addition,
and because it would put a GPU dependency in the path of scheduler tests.

**CUDA first, CPU later or never.** Rejected. Without a trusted reference
implementation there is nothing to compare kernel output against, so every
numerical discrepancy becomes an open-ended investigation. The CPU path is also
what allows the test suite to run in CI without GPU runners.

## Consequences

**Good.** The scheduler, cache, tokenizer and API are all developed and tested
with no GPU. The CPU backend becomes the correctness oracle for kernel tests
later. Adding CUDA touches `orion-cuda` and the backend selection, nothing else.

**Bad.** CPU inference is slow and will never be the production path, so some
of that work is scaffolding. The trait may need widening once real CUDA
requirements appear — memory pools and streams in particular are not modelled
yet.

**Honest limitation.** No CUDA code has been written or verified. The
`orion-cuda` crate is a placeholder. Nothing in this repository claims GPU
acceleration, and no GPU benchmark numbers exist because no GPU has been
available to produce them.
