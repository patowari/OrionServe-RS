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

### Not yet implemented

Model execution, tokenizer integration, the HTTP API, observability
exporters, CUDA, quantization and distributed inference are all `planned`. No
benchmarks have been run and no performance claims are made.
