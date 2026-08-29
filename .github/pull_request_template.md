## What this changes

<!-- One or two sentences. What behaviour is different afterwards? -->

## Why

<!-- The problem this solves. Link an issue if there is one. -->

## How it was verified

<!-- Which tests cover this? If it is a performance change, what was measured,
     on what hardware, before and after? -->

## Checklist

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] New behaviour has tests
- [ ] Design docs updated if a documented decision changed
- [ ] Performance claims include a reproducible measurement and hardware details
