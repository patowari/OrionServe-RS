# Benchmarks

Two tools, measuring different things. See [docs/benchmarking.md](../docs/benchmarking.md)
for the methodology and what each metric means.

## Microbenchmarks

Measure the data structures this project owns — block allocation, scheduling,
prefix hashing, sampling, CPU tensor primitives. No GPU or model required.

```bash
cargo bench -p orion-bench
```

Criterion writes HTML reports to `target/criterion/`.

These exist to catch regressions in code this project controls. They are **not**
an inference benchmark: nobody's request latency is decided by block-table
lookup speed.

## Load testing

Drives a running server over HTTP and measures what a client experiences.

```bash
# Start the server.
orion serve --model /path/to/model &

# Sweep concurrency across all four workload shapes.
orion-bench --url http://127.0.0.1:8000 \
            --concurrency 1,10,50,100 \
            --requests 200

# Or one shape.
orion-bench --workload long_prompt_short_output --concurrency 50
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:8000` | Server base URL |
| `--concurrency` | `1,10,50` | Concurrency levels to sweep |
| `--requests` | `50` | Requests per level |
| `--warmup` | `5` | Warm-up requests, discarded |
| `--workload` | all four | `short_prompt_short_output`, `short_prompt_long_output`, `long_prompt_short_output`, `long_prompt_long_output` |
| `--precision` | `f32` | Recorded in metadata; the harness cannot detect what the server loaded |
| `--notes` | — | Free text stored with the results |

### Output

`benchmarks/results/results-<timestamp>.json` — complete, with hardware
metadata. `benchmarks/results/results-<timestamp>.csv` — one row per
configuration, for trend tracking.

Results are gitignored by default. Commit a specific run only when it is being
cited, and only with its metadata intact.

## Rules this harness enforces

- **Warm-up is discarded** — the first requests absorb page faults, allocator
  growth and a cold prefix cache.
- **Failures are counted, never dropped** — a fast-failing server is not fast.
- **Prompts vary per request** — identical prompts would measure the prefix
  cache rather than the engine.
- **CPU-only runs say so** — the summary prints
  `GPU: none detected - THIS IS A CPU-ONLY RESULT`, so a number cannot be
  quoted out of context by accident.
- **Percentiles use nearest-rank** — interpolation invents values that were
  never observed.

## Current results

**None published.** The development machine has no NVIDIA GPU, so no serving
benchmark worth publishing has been run. Microbenchmark figures appear in
[docs/performance-journal.md](../docs/performance-journal.md) with the machine
they were measured on.

No comparison against vLLM, TGI or TensorRT-LLM is published: a fair comparison
needs GPU hardware this project has not run on, and a CPU-only figure beside
their GPU numbers would be meaningless.
