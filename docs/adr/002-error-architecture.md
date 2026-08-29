# ADR 002: Three-way error classification

**Status:** Accepted
**Date:** 2026-08-29

## Context

An inference server fails in ways that demand different responses. A malformed
`config.json` should stop the process at startup. A prompt exceeding the context
window should return 400 to one client. A full KV cache should return 503 with
`Retry-After` — the same request may well succeed a second later. An internal
invariant violation should be logged in full and returned as an opaque 500.

Collapsing these into one error type forces the HTTP layer to decide status
codes by inspecting messages, which is fragile and silently wrong the moment a
message is reworded.

## Decision

Three error enums in `orion-core::error`, split by *when* the failure occurs
and *who* can act on it:

- `ConfigError` — operator misconfiguration. Fatal at startup. Names the
  offending field.
- `ModelError` — missing, malformed, or unsupported model artifact. Fatal at
  load, never at request time.
- `EngineError` — per-request failure.

`EngineError` carries classification methods rather than leaving it to callers:

- `is_client_error()` — caused by the caller's input (4xx)
- `is_retryable()` — a transient capacity signal (429/503)
- `code()` — a short stable string exposed in API bodies so clients can branch
  without parsing prose

## Alternatives considered

**One error enum for everything.** Rejected: per-request code paths could then
return "model file corrupt", which is meaningless to a client and impossible to
handle. The type system should make that unrepresentable.

**`anyhow::Error` throughout.** Excellent for applications, wrong for a library
boundary. Rejected because the API layer genuinely needs to match on failure
kind to choose a status code, and `anyhow` erases exactly that.

**HTTP status codes embedded in the error type.** Rejected because it puts
transport concerns in the domain crate; `orion-core` would then need an opinion
about HTTP, and a gRPC surface would have to map backwards out of it.

**A `retryable: bool` field on a single variant-less error struct.** Rejected:
the structured variants carry useful diagnostic data (`needed`/`available`
blocks, `prompt_tokens`/`context_len`) that a flat struct would either lose or
stringify.

## Consequences

**Good.** Status-code mapping is total and testable without string matching.
Adding a variant forces every match site to be revisited. Diagnostic detail
survives to the log line without being pasted into the client-facing message.

**Bad.** Three types mean conversion boilerplate at boundaries; `OrionError`
exists purely to unify them for the binary's top level. Deciding which enum a
new failure belongs to is occasionally a judgement call.

**Deliberate.** `EngineError::Internal` is never shown to clients beyond an
opaque 500. Internal state — block ids, sequence ids, invariant descriptions —
is useful to operators and is exactly what should not leak to callers.
