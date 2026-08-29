# Distributed Inference

## Status

**Not implemented, and not verified.** The development machine has no NVIDIA
GPU at all, let alone several, and no NCCL installation.

What exists: the *partitioning calculus* in `crates/orion-distributed` — the
arithmetic deciding how weights split across ranks, which is verifiable without
hardware and is tested. Everything that touches a device is a stub.

No scaling efficiency is claimed. Scaling efficiency is a measurement.

## Tensor parallelism

Each linear layer is split across GPUs; each holds a slice of the weights, and a
collective stitches the partial results together.

Unlike pipeline parallelism it introduces no bubble and keeps every GPU busy on
every token — at the cost of a collective on the critical path of every layer,
which is why interconnect performance decides whether it is worth doing.

### Where the splits go

```text
        x  (replicated on every rank)
        |
   +----+----+   column-parallel: split the OUTPUT dimension
   v         v   no communication needed; each rank has all of x
  W_a       W_b
   |         |
   v         v   partial outputs, each [tokens, out/N]
  y_a       y_b
   |         |
   +----+----+   row-parallel: split the INPUT dimension
        v        each rank produces a partial sum
    AllReduce    <-- one collective
        |
        v
        y
```

Pairing a column-parallel layer with a row-parallel one is the key move: rank
`i`'s slice of the first output is exactly the input its slice of the second
weight needs, so the intermediate never has to be gathered.

That gives **one** AllReduce per attention block and one per MLP block, rather
than two of each. For a 32-layer model that is 64 collectives per token instead
of 128.

### Attention

Split by **head**. Each rank owns whole heads, so the softmax — which must see a
complete head — needs no communication.

- Q, K, V projections: column-parallel, split by head
- Output projection: row-parallel
- One AllReduce after the output projection

**The KV head constraint.** Grouped-query attention models have few KV heads:
Llama-3-8B has 8. Splitting beyond that would require replicating them, which is
not implemented. `ParallelLayout::compute` reports this as a configuration error
rather than silently choosing a layout the operator did not ask for.

This is a real limitation and it binds sooner than people expect. An 8-KV-head
model cannot use 16-way tensor parallelism under this design.

### MLP

- `gate_proj`, `up_proj`: column-parallel
- `down_proj`: row-parallel
- One AllReduce after `down_proj`

The SwiGLU activation is elementwise, so it needs no communication — each rank
activates its own slice.

### What is not split

Norms and biases are replicated. They are tiny, and splitting them would cost
more in communication than it saves in memory.

## The KV cache saving

The reason tensor parallelism is attractive for long context, beyond fitting
larger models: each rank stores only its own KV heads, so cache memory per GPU
falls linearly with world size.

For Llama-3-8B at FP16, 128 KiB per token becomes 32 KiB per GPU at 4-way
parallelism — often the difference between fitting a long-context workload and
not.

`ParallelLayout::local_kv_bytes_per_token` computes this, and it is tested.

## Communication cost

Two AllReduces per layer, each moving `tokens x hidden_size` elements. A ring
AllReduce moves roughly `2(N-1)/N` times the buffer per rank.

For Llama-3-8B at FP16 with 2-way parallelism that is on the order of hundreds
of kilobytes per token across 64 collectives.
`CommunicationEstimate::for_model` computes it from the layer structure.

**This is an estimate, not a measurement.** It is useful for deciding whether
tensor parallelism is worth attempting on a given interconnect before writing
any of it: if the estimate says collectives will dominate, they will.

The practical consequence is that **latency, not bandwidth, usually dominates**.
The buffers are small and there are many of them, so 64 round trips per token
matters more than the total bytes. NVLink and PCIe differ far more in latency
for this pattern than their bandwidth figures suggest.

## Why NCCL

Collectives sit behind the `Collective` trait rather than calling NCCL directly,
so the partitioning logic can be tested against a single-rank no-op with no GPU.
That is the only implementation that currently exists.

NCCL is the intended backing implementation because it handles topology
detection, ring and tree algorithm selection, and NVLink/PCIe/InfiniBand
transport. Reimplementing that would be a project in itself, and would be worse.

`SingleRank` is not a placeholder — it is the *correct* implementation for
`world_size == 1`. Having one code path rather than two means every single-GPU
test exercises the distributed path.

## Scaling efficiency

The number that matters, once this runs:

```
efficiency = throughput(N GPUs) / (N x throughput(1 GPU))
```

It is always below 1. The gap is communication overhead plus whatever load
imbalance the split introduces.

**No efficiency figure will be published without a measurement**, on stated
hardware, with the interconnect named. A tensor-parallel number on NVLink and
the same number over PCIe are different results, and reporting one as the other
would be dishonest.

## Speculative decoding

Related, and also unimplemented.

```text
Small draft model
       |  generates k candidate tokens cheaply
       v
Large target model
       |  verifies all k in ONE forward pass
       v
Accepted tokens (0..k)
```

The win comes from the target model verifying `k` tokens in a single pass — the
same memory traffic as generating one. If the draft model is right most of the
time, several tokens come out per target-model pass.

The catch is that acceptance rate decides everything, and it is entirely
workload dependent. A draft model that agrees 80% of the time is transformative;
one that agrees 30% of the time is slower than not bothering, because the draft
model's own cost is pure overhead on every rejection.

That is why it stays unimplemented rather than half-implemented: there is no
point building it without the ability to measure acceptance rate on real
traffic.

## Not designed here

**Pipeline parallelism** — splitting layers rather than within layers. Lower
communication, but introduces a bubble and only helps when a model does not fit
even when split.

**Expert parallelism** — for mixture-of-experts models, which this engine does
not support.

**Multi-node** — everything above assumes one machine. Crossing nodes changes
the latency budget by an order of magnitude and would need a different design.
