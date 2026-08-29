# ADR 006: Single-threaded engine step loop

**Status:** Accepted
**Date:** 2026-08-29

## Context

The API layer is asynchronous and handles many concurrent connections. The
engine — scheduler plus KV cache manager plus model forward pass — has to
consume from it.

The scheduler and cache manager together hold one consistent view of which
sequence owns which block. Allocation decisions depend on scheduling decisions
and vice versa: whether to admit a sequence depends on free blocks, and freeing
blocks depends on which sequences were preempted.

## Decision

The engine step loop runs on a single thread. `KvCacheManager` is **not**
internally synchronized and is mutated through `&mut self`. Async request
handlers communicate with it over **bounded** channels.

## Alternatives considered

**`Arc<Mutex<KvCacheManager>>` shared across async tasks.** The obvious
approach, and rejected deliberately. It invites callers to interleave
allocation decisions with scheduling decisions — check free blocks, yield,
allocate — which is precisely the race that produces double-allocation and
use-after-free bugs in a block allocator. The lock would also be taken on every
allocation, so contention would rise exactly when the server is busiest.

**Fine-grained locking per block or per sequence.** Rejected: allocation
decisions are global (how many blocks are free?), so per-object locks do not
help and introduce lock-ordering hazards.

**Sharding the cache across threads.** Rejected as premature. It would fragment
the free pool — the thing paging exists to avoid — for a bottleneck that has
not been measured and is not expected to exist.

**Unbounded channels between API and engine.** Rejected. Unbounded queues
convert overload into memory exhaustion. Bounded channels provide backpressure,
which the API layer turns into a 429.

## Consequences

**Good.** Cache accounting has one owner and one mutation path. An entire class
of concurrency bug is unrepresentable rather than merely tested for. `&mut self`
means the borrow checker enforces exclusivity at compile time.

**Bad.** Scheduling and sampling cannot overlap with each other. This is
acceptable because the engine is bound on the forward pass, not on block
arithmetic — but it is an assumption, and it is unverified until end-to-end
profiling exists. If scheduling ever shows up in a profile, the fix is to move
work *out* of the loop, not to make the loop concurrent.

**Constraint accepted.** Multi-GPU tensor parallelism will need several worker
threads executing one logical step. That design keeps a single scheduling
authority and fans out only the forward pass, so this decision stands; it is
recorded here so the eventual distributed work does not silently violate it.
