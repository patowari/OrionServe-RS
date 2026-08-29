# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cargo workspace of eleven crates, split so the scheduler and KV cache depend
  on no GPU, model, or async runtime.
- `orion-core`: three-way error classification (config / model / engine) with
  retryability and client-error predicates; newtype identifiers; validated
  engine configuration; validated sampling parameters; sequence lifecycle state
  machine; `Backend`, `LanguageModel` and `Sampler` trait boundaries.
- `orion-kv-cache`: reference-counted paged block pool with FIFO reclamation
  doubling as LRU; per-sequence block tables; chained-hash prefix caching with
  token-equality verification on lookup; failure-atomic allocation.
- `orion-models`: HF `config.json` normalization across architectures,
  memory-mapped safetensors loading, CPU tensor primitives (RMSNorm, RoPE,
  SwiGLU, linear), paged attention that gathers K/V through a block table, and
  a decoder-only transformer implementing `LanguageModel`.
- `orion-tokenizer`: HF tokenizer wrapper, incremental streaming decode that
  never emits a partial character, chat templates for Llama 3 and ChatML, and
  a stop-sequence matcher that works across token boundaries.
- `orion-runtime`: the engine loop on a dedicated thread, driving scheduler,
  model and sampler; bounded command channel for backpressure; automatic
  cancellation when a client disconnects.
- `orion-api`: OpenAI-compatible `/v1/chat/completions` and `/v1/completions`
  with SSE streaming, plus `/health`, `/ready` and `/v1/models`.
- `orion-observability`: Prometheus metrics with purpose-chosen histogram
  buckets, and structured logging in pretty or JSON form.
- `orion-cli`: `orion serve` and `orion inspect`, with graceful shutdown on
  SIGINT and SIGTERM.
- 20 end-to-end tests over the real HTTP surface, covering streaming/non-
  streaming agreement, concurrency, prefix-cache reuse, error mapping and a
  KV-block leak check.
- `orion-runtime`: sampling engine — sign-aware repetition penalty, temperature,
  top-k via O(n) selection, nucleus (top-p) filtering, numerically stable
  softmax, and per-request seeded RNG so a seeded request is reproducible
  regardless of concurrent load.
- `orion-scheduler`: continuous batching; chunked prefill; preemption by
  recompute with front-of-queue recovery; admission control; per-step timeout
  expiry; `FakeCache` test double for forcing allocation failures.
- Design documents for the architecture, KV cache and scheduler; six ADRs.
- CI running formatting, clippy, tests, release build, MSRV check, security
  audit and documentation build.

### Fixed

- Scheduler: a sequence part-way through chunked prefill belonged to neither
  the decode path nor the admission path and stalled permanently. Resolved by
  resuming partial prefills before new admissions.
- KV cache: recycling a cached block left its prefix-cache entry in place, so a
  later lookup could adopt a block whose contents had been overwritten.
  `BlockPool::allocate` now reports the invalidated hash.
- KV cache: eviction stopped at the first uncached free block instead of
  skipping it, so a cached block behind it could never be evicted.

### Fixed (continued)

- Tokenizer: `IncrementalDecoder::finish` required the full decode to start
  with everything already emitted, which silently dropped the tail whenever a
  character had been held back mid-stream. It now keys off the longest common
  prefix.
- Tokenizer: window re-anchoring reset the emitted-byte offset to zero instead
  of rebasing it, duplicating the retained tail ("helllo wworlld").

### Not yet implemented

CUDA, quantization and distributed inference are `planned`. No benchmarks have
been run and no performance claims are made.
