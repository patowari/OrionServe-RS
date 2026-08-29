# Security Policy

## Reporting a vulnerability

Report security issues privately through GitHub's [private vulnerability
reporting](https://github.com/patowari/orionserve-rs/security/advisories/new).
Please do not open a public issue for a vulnerability.

Include: what the issue is, how to reproduce it, and what an attacker could
achieve. You should get an acknowledgement within a few days.

## Threat model

OrionServe-RS is a network-facing server that runs untrusted input through a
model. The interesting attack surfaces are:

**Resource exhaustion.** Prompt length, generated tokens, request body size,
queue depth, and concurrency are all bounded by configuration, and requests are
rejected at admission rather than accepted and starved. Unbounded growth in any
of these is a vulnerability, not just a bug.

**Model artifacts.** A model directory is trusted input from the operator, not
from clients. Malformed checkpoints must fail cleanly rather than crash or
execute anything. Path handling must not escape the configured model directory.

**Information disclosure.** Internal state — block ids, sequence ids, invariant
messages — must never appear in an API response. `EngineError::Internal` is
logged in full and returned as an opaque 500 by design. Secrets must not be
logged.

**Cross-request leakage.** Prefix caching shares KV blocks between requests.
That sharing is only sound when the entire preceding token history matches,
which the chained block hash enforces and a token-equality check verifies on
every lookup. A bug allowing one request to read another's cached state would
be a serious vulnerability; see `docs/kv-cache.md` for the guards.

## Supported versions

Pre-1.0. Only `main` receives fixes.
