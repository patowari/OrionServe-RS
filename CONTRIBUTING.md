# Contributing to OrionServe-RS

## Quality gates

Every change must keep these green. CI enforces all four.

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo build --workspace --release
```

## What good changes look like

**New behaviour comes with tests.** Particularly in `orion-scheduler` and
`orion-kv-cache`, where bugs are hard to reproduce after the fact. Both crates
run without a GPU, so there is no excuse for an untested policy change.

**Performance claims come with measurements.** A statement that something is
faster needs: the benchmark used, the hardware it ran on, and before/after
numbers. Record it in `docs/performance-journal.md`. Estimated or extrapolated
figures are not acceptable — if it was not measured, say so.

**Design changes come with reasoning.** If a change contradicts an ADR in
`docs/adr/`, write a superseding ADR rather than quietly editing the old one.
The reasoning behind a decision that turned out wrong is more useful than a
tidy record pretending it never happened.

**Unimplemented things are marked `planned`.** Never document a feature as
working before it does. The README's feature table is authoritative and must
stay accurate.

## Commit messages

Semantic prefixes, scoped to a crate or subsystem:

```
feat(scheduler): resume partially prefilled sequences
fix(kv-cache): invalidate index entry when a cached block is recycled
perf(kv-cache): avoid free-list scan on the allocation path
docs(adr): record the single-threaded engine loop decision
test(scheduler): pin starvation freedom under repeated preemption
```

The body should say *why*, not restate the diff.

## Code style

- No `unsafe` without an isolated module, a documented safety invariant, and a
  comment explaining why the safe alternative was insufficient.
- Prefer `&mut self` over interior mutability. If something needs a lock, the
  comment should say what race it prevents.
- Errors get structured variants, not stringified context.
- Comments explain reasoning, not mechanics. `// increment the counter` is
  noise; `// front, not back: otherwise arrivals starve preempted sequences`
  is the reason the line exists.

## Review

Assume the reviewer will ask: Is this allocation necessary? Can this race? What
happens under OOM? Can the request be cancelled? Is memory released on every
path? Is there a test? Is there a benchmark? Answering those in the PR
description saves a round trip.
