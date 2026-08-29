# ADR 001: Multi-crate Rust workspace

**Status:** Accepted
**Date:** 2026-08-29

## Context

OrionServe-RS spans concerns that have very different dependency needs: HTTP
serving wants an async runtime, model execution wants tensor libraries and
eventually CUDA, and the scheduler and cache allocator want neither. Building
it as a single crate would mean every test run links everything, and a CUDA
feature flag would leak into code that has nothing to do with GPUs.

The engine also has two components — the scheduler and the KV cache allocator —
where bugs are both most likely and most expensive: they are concurrent,
stateful, and their failure modes (memory leaks, starvation, use-after-free of
blocks) are hard to reproduce and hard to attribute. Whatever structure is
chosen has to make those two components trivially testable.

## Decision

A Cargo workspace of eleven crates, split by **what can be tested without
what** rather than by subject matter.

The load-bearing constraint: `orion-core` depends only on `serde`, `thiserror`
and `uuid`; `orion-kv-cache` and `orion-scheduler` depend only on `orion-core`.
Neither pulls in an async runtime, a tensor library, or a GPU.

Dependency versions are centralized in `[workspace.dependencies]` so crates
cannot drift onto different versions of the same library.

## Alternatives considered

**Single crate with feature flags.** Simplest to start. Rejected because
`cargo test` would compile the entire dependency tree — including, eventually,
CUDA bindings — to run a scheduler unit test. Feature-gating also tends to
produce combinations that are never built and quietly stop compiling.

**Two crates (library + binary).** The conventional Rust split. Rejected
because it does not achieve the goal: the scheduler would still sit in the same
crate as the model runtime and inherit its dependencies.

**Separate repositories.** Rejected outright. Cross-repo changes to a trait and
its implementations would need coordinated releases, for a project with one
deployment target.

## Consequences

**Good.** The scheduler and cache test suites run in milliseconds on any
machine, with no GPU and no model. A contributor can work on scheduling policy
without a CUDA toolkit. The `Backend` abstraction has a natural home, and the
CUDA crate can be added without touching anything above it.

**Bad.** More boilerplate: eleven `Cargo.toml` files, and adding a dependency
means touching two. Cross-crate refactors are wider. `cargo build` of the full
workspace is slower than a single crate would be, though incremental builds of
one crate are much faster.

**Accepted cost.** A trait needed by two crates that should not depend on each
other has to live in `orion-core`, even when it is not conceptually "core" —
`KvCacheManagerLike` is defined there so `orion-scheduler` can be generic over
the cache without depending on `orion-kv-cache` at all. This is slightly
awkward and is the price of the dependency direction.
