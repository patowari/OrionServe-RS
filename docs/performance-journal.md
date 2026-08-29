# Performance Journal

Every optimization, with the measurement that motivated it and the measurement
that justifies it. Entries are append-only: an optimization that turned out not
to help stays in the record, because knowing what was tried and rejected is as
useful as knowing what worked.

**Rule:** no entry without before/after numbers from an actual run, and the
machine they came from.

---

## Reference machine

Unless an entry says otherwise, measurements come from:

```
OS:        Windows 11 Pro 10.0.26200
CPU:       8 logical cores (x86_64)
GPU:       none — no NVIDIA device, no CUDA toolkit
Rust:      1.97.1 (stable), release profile, lto = "thin", codegen-units = 1
Harness:   criterion 0.8, 100 samples, 1s warm-up, 2s measurement
```

**These are CPU-only numbers.** No GPU has been available to this project, so
no entry here describes GPU performance.

---

## 001 — Sampling: skip masked entries in softmax and top-p

**Date:** 2026-08-29
**Component:** `orion-runtime::sampling`

### Problem

The first microbenchmark run showed sampling dominating the decode path:

```
sampling/greedy/128256          115.92 µs
sampling/top_k_top_p/128256    9776.20 µs      <-- 84x slower than greedy
```

9.8 ms per sampled token is not a rounding error next to a forward pass — it
would be the single largest cost in a decode step, and it scales with
vocabulary rather than with anything the caller controls. A Llama-3 vocabulary
is 128,256 tokens, so this is the realistic case, not a corner.

### Hypothesis

The filters were operating over the *entire* vocabulary even though top-k had
already reduced the live candidate set to `k` (50 in the benchmark). Two
specific suspects:

1. `apply_top_p` built and sorted an index permutation of all 128,256 entries —
   O(n log n) with indirect comparisons that defeat the cache — when at most
   `k` of them could possibly be inside the nucleus.
2. `softmax` called `exp()` on all 128,256 entries, including the ~128,200 at
   the `MASKED` sentinel. Each is a transcendental call producing a value that
   underflows to zero.

### Implementation

Two changes, measured separately so each could be attributed.

**Change A — `apply_top_p` sorts only live candidates.** Collect `(index,
probability)` pairs for unmasked entries, sort that, and mask the tail. A
masked logit has zero probability and can never be in the nucleus, so excluding
it changes nothing observable.

**Change B — `softmax` skips masked entries.** Entries at or below `MASKED`
are left at zero instead of being exponentiated. The `max <= MASKED` guard
retains the existing uniform fallback for a fully-masked row.

### Results

Change A alone:

| Benchmark | Before | After | Change |
|---|---|---|---|
| `top_k_top_p/32000` | 1.950 ms | 1.670 ms | −14.3% |
| `top_k_top_p/128256` | 9.776 ms | 6.661 ms | −31.9% |

Change B, on top of A:

| Benchmark | Before (A) | After (A+B) | Change |
|---|---|---|---|
| `top_k_top_p/32000` | 1.670 ms | 0.282 ms | **−82.8%** |
| `top_k_top_p/128256` | 6.661 ms | 0.963 ms | **−85.5%** |

Combined, from the original baseline:

| Benchmark | Before | After | Speedup |
|---|---|---|---|
| `top_k_top_p/32000` | 1.950 ms | 0.282 ms | **6.9×** |
| `top_k_top_p/128256` | 9.776 ms | 0.963 ms | **10.2×** |

Greedy sampling moved within noise (`p = 0.79` and `p = 0.20`), as expected —
it takes an early return before either filter.

### Interpretation

Change B dominating Change A is the interesting result, and it inverts the
obvious intuition. The `O(n log n)` sort *looks* like the expensive operation
next to an `O(n)` exponentiation pass, and that is where I looked first. But
`exp` is a transcendental costing tens of cycles, applied 128,256 times, while
the sort's comparisons are on a small live set once the permutation is skipped.
Asymptotic reasoning pointed at the wrong line; the measurement pointed at the
right one.

The general lesson the benchmark taught: the sampler was written as a pipeline
of independent filters, each correct in isolation, each unaware that a previous
stage had already eliminated 99.96% of its input.

### Tradeoffs

- **Branch in a hot loop.** `softmax` now tests each element before
  exponentiating. When *nothing* is masked (top-k disabled, top-p at 1.0) that
  branch is pure overhead — but in that configuration `apply_top_p` returns
  immediately and softmax is only reached by the sampler's final call, so the
  cost is one predictable, well-predicted branch per element.
- **An allocation in `apply_top_p`.** The candidate vector is `k` entries
  rather than `n`, so it is far smaller than the index permutation it replaced.
- **`MASKED` is now load-bearing in two more places.** It was already the
  sentinel; it is now also a filter predicate. The constant's doc comment
  explains why it is a large finite negative rather than `-inf`, and that
  reasoning now guards more code.

### Correctness

All 34 sampling tests pass unchanged, including:

- `top_p_always_keeps_at_least_one_token`
- `stochastic_sampling_respects_a_top_k_of_one`
- `softmax_of_fully_masked_logits_does_not_produce_nan`
- `the_same_seed_produces_the_same_sequence`
- `sampling_reflects_the_distribution` (statistical, 3000 draws)

Seeded reproducibility passing matters most here: it shows the *sequence of
sampled tokens* is byte-identical to before, not merely that the code is
faster.

---

## Baseline measurements

Recorded 2026-08-29 on the reference machine, after entry 001.

### KV cache

| Operation | Time |
|---|---|
| `allocate_and_free` (128-token prompt) | 1.61 µs |
| `allocate_and_free` (1024-token prompt) | 3.21 µs |
| `allocate_and_free` (4096-token prompt) | 6.15 µs |
| `append_token` (decode-step path) | 364 ns |

Allocation scales sub-linearly with prompt length: 32× the tokens costs 3.8×
the time, because per-block work dominates the per-call overhead only once
prompts are large.

`append_token` at 364 ns is the number that matters most — it runs once per
sequence per step. At 256 concurrent sequences that is 93 µs per step of pure
cache bookkeeping.

### Prefix cache

| Operation | Time |
|---|---|
| `hash_block` (16 tokens) | 13.1 ns |
| `hash_block` (64 tokens) | 75.5 ns |
| `hash_block` (256 tokens) | 333 ns |
| `hash_2048_token_prompt` (chained, 16-token blocks) | 2.94 µs |
| `allocate_with_cache_hit` (512-token prompt) | 86.2 µs |

Hashing is linear in tokens, ~1.3 ns per token, which is negligible next to the
prefill it can avoid.

`allocate_with_cache_hit` at 86 µs looks expensive relative to a cold
`allocate`, and is a known follow-up: it is dominated by the benchmark's setup
of a warm cache per batch rather than by the lookup itself. Attributing it
properly needs a finer benchmark.

### Scheduler

| Operation | Time |
|---|---|
| `decode_step` (8 sequences) | 130 µs |
| `decode_step` (64 sequences) | 166 µs |
| `decode_step` (256 sequences) | 276 µs |

Scaling from 8 to 256 sequences (32×) costs 2.1× the time, so the per-sequence
marginal cost is small and a fixed overhead dominates at low counts. That is
the right shape for a scheduler: it means batch size is not limited by
scheduling cost.

### CPU tensor primitives

| Operation | Shape | Time |
|---|---|---|
| `rms_norm` | 8 × 512 | 4.52 µs |
| `linear` | 8 × 512 × 512 | 1.60 ms |
| `swiglu` | 8 × 512 | 17.1 µs |

**These are reference-implementation numbers and are deliberately not
optimized.** `linear` is a naive triple loop with no blocking, no SIMD
intrinsics and no BLAS. It exists to be *obviously correct* so CUDA kernels can
be validated against it. Optimizing it would be work spent on a path that will
never be the production one.

---

## Planned investigations

Recorded so the reasoning is not lost, not as claims:

- **`allocate_with_cache_hit` attribution.** Separate the lookup cost from the
  benchmark's per-batch cache warm-up.
- **Block size.** `docs/kv-cache.md` argues for 16 from first principles. It
  has not been measured against a real workload, and should be.
- **Scheduler fixed overhead.** 130 µs at 8 sequences suggests a constant cost
  worth finding, though it is currently far below forward-pass time.
- **Prefix cache hit rate under realistic traffic.** The mechanism is tested;
  its value depends entirely on workload and has not been measured end to end.
