# Architecture Decision Records

Each ADR records one decision: the context that forced it, what was chosen,
what was rejected, and what the choice costs.

They are written when the decision is made and are not revised afterwards. A
decision that turns out badly gets a *new* ADR superseding the old one, because
the reasoning that led to a wrong choice is more useful than a tidy record that
pretends it never happened.

| # | Decision | Status |
|---|---|---|
| [001](001-rust-workspace.md) | Multi-crate Rust workspace | Accepted |
| [002](002-error-architecture.md) | Three-way error classification | Accepted |
| [003](003-paged-kv-cache.md) | Paged block-based KV cache | Accepted |
| [004](004-scheduler-architecture.md) | Continuous batching, decode-first | Accepted |
| [005](005-backend-abstraction.md) | Backend trait, CPU first | Accepted |
| [006](006-single-threaded-engine-loop.md) | Single-threaded engine step loop | Accepted |
