# Benchmarking

## The problem with inference benchmarks

Most published LLM serving numbers are not comparable with each other. A
"tokens per second" figure is meaningless without knowing:

- what hardware it ran on
- what model, at what precision
- how long the prompts were, and how long the outputs
- how many requests were in flight
- whether the number counts prompt tokens, output tokens, or both
- whether failed requests were counted or quietly dropped

Change any one of those and the number moves by an order of magnitude. This
project's harness records all of them with every result, and refuses to print a
figure without them.

## Two kinds of measurement

These are deliberately separate, and conflating them is the most common way
benchmark claims become dishonest.

### Microbenchmarks — `cargo bench -p orion-bench`

Measure the data structures this project *owns*: the block allocator, the
scheduler pass, prefix hashing, sampling. None touch a device, so they are
meaningful on any machine.

**What they are for:** catching regressions. A scheduler pass that becomes 10x
slower is a real problem even when the forward pass still dominates wall clock,
because it means an algorithmic mistake has crept in.

**What they are not:** an inference benchmark. Nobody's request latency is
determined by how fast a block table lookup is.

### Load testing — `orion-bench`

Drives real HTTP requests against a running server and measures what a client
experiences. This is the only thing that can honestly be called serving
performance.

```bash
orion serve --model /path/to/model &
orion-bench --url http://127.0.0.1:8000 \
            --concurrency 1,10,50,100 \
            --requests 200
```

## Metrics, and what each one actually tells you

### Time to first token (TTFT)

Arrival to first generated token. Dominated by **queueing delay plus prefill**.

Measured from arrival, not from when the scheduler picked the request up.
Queueing delay is latency the client experiences, and excluding it is the
easiest way to make a saturated server look fast.

Rises sharply when: the queue is deep, prompts are long, or a large prefill is
monopolizing steps (which is what chunked prefill exists to prevent).

### Time per output token (TPOT)

Mean gap between successive tokens after the first. This is what a streaming
client perceives as "speed" — it determines whether text appears smoothly or in
stutters.

Computed as `(total - ttft) / (tokens - 1)`, and **undefined below two tokens**
rather than reported as zero. A zero would drag an average down and hide the
fact that nothing was measured.

Rises when: batch size grows (more work per step), or the KV cache saturates
and preemption starts.

### End-to-end latency

Arrival to last token. Roughly `TTFT + TPOT × output_tokens`, so it is mostly
determined by output length. Useful for non-streaming clients; misleading as a
headline number, because a request that generates 1000 tokens will always look
slower than one generating 10.

### Throughput

Two figures, reported separately:

- **Output token throughput** — generated tokens per second. The number that
  matters for capacity planning.
- **Request throughput** — completed requests per second. Depends entirely on
  output length, so only comparable within one workload shape.

Both count **only successful requests**. A server that fails fast is not fast.

### Percentiles

p50, p90, p95, p99 from the sorted sample, using **nearest-rank**. Interpolated
percentiles invent values that were never observed, which at realistic sample
sizes makes p99 a fiction.

p99 matters more than the mean: it is the experience of the unluckiest 1% of
requests, and in a system with preemption and queueing the tail is where the
scheduler's behaviour actually shows up.

## Workload shapes

One number across a mixed workload hides everything interesting, because
prompt and output lengths stress completely different parts of the engine.

| Shape | Prompt | Output | Stresses | Resembles |
|---|---|---|---|---|
| short / short | 128 | 128 | scheduling overhead | chat |
| short / long | 128 | 1024 | decode batching, cache growth | generation |
| long / short | 2048 | 128 | prefill, chunking | summarization, RAG |
| long / long | 2048 | 1024 | everything, cache pressure worst | agents |

The long-prompt shapes are where chunked prefill and preemption earn their
keep. A benchmark that only runs short/short will never exercise them.

## Methodology rules this harness enforces

**Warm-up is discarded.** The first requests absorb page faults, allocator
growth and a cold prefix cache. Including them either flatters or penalizes a
run depending on where the noise lands.

**Failures are counted, not dropped.** A run that reports high throughput while
half the requests errored is worse than useless. Failures appear in the summary
and are excluded from throughput.

**Prompts vary between requests.** Sending the identical prompt N times would
measure the prefix cache, not the engine. The harness prefixes each request
with its index.

**CPU-only runs are labelled.** If `nvidia-smi` reports nothing, the summary
says `GPU: none detected - THIS IS A CPU-ONLY RESULT` in the output itself, so
a figure cannot be quoted out of context by accident.

**Hardware is recorded automatically.** OS, architecture, core count, GPU name,
CUDA version and driver version go into every JSON result.

## Comparing against other engines

A fair comparison against vLLM, TGI, llama.cpp or TensorRT-LLM requires:

1. the same GPU, driver and CUDA version
2. the same model and precision
3. the same prompt and output length distributions
4. the same concurrency
5. the same measurement point (client-side, including queueing)
6. both systems warmed up identically

**No such comparison is published in this repository**, because the project has
not been run on GPU hardware. Publishing a CPU-only figure next to another
engine's GPU numbers would be meaningless, and doing so deliberately would be
dishonest.

When GPU work lands, comparisons will be run under the conditions above and the
full configuration of *both* systems will be recorded.

## Output

Results are written to `benchmarks/results/` as JSON (complete, including
metadata) and CSV (one row per configuration, for trend tracking).

```
workload,concurrency,ok,failed,output_tokens,req_per_s,out_tok_per_s,
ttft_p50_ms,ttft_p99_ms,tpot_p50_ms,latency_p50_ms,latency_p99_ms,gpu
```

## Profiling

When a benchmark shows a regression, these find the cause:

**CPU:** `cargo flamegraph --bin orion -- serve --model ...` under load.

**Tracing:** `RUST_LOG=orion=debug` emits per-step timings. The
`orion_step_duration_seconds` histogram separates scheduling cost from forward
pass cost.

**GPU (once CUDA lands):** Nsight Systems for the timeline —
`nsys profile ./target/release/orion serve ...` — and Nsight Compute for
individual kernels. `Backend::synchronize` exists so that GPU timings are
honest: without it, an asynchronous launch queue makes kernels appear
instantaneous.

## Reproducing a published result

Every JSON result contains the full metadata needed to reproduce it. To repeat
a run:

```bash
# Read the recorded configuration.
jq '.[0].metadata' benchmarks/results/results-<timestamp>.json

# Re-run with the same shape and concurrency.
orion-bench --url http://127.0.0.1:8000 \
            --workload long_prompt_short_output \
            --concurrency 50 \
            --requests 200 \
            --notes "reproducing results-<timestamp>"
```

If a result cannot be reproduced from what is recorded with it, that is a bug
in the harness.
