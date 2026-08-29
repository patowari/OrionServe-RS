# KV Cache Design

## Why a KV cache exists at all

A decoder-only transformer generating token *t* must attend to every token
before it. Without caching, producing token *t* means recomputing the key and
value projections for all *t-1* preceding tokens — turning generation of *n*
tokens into O(n²) work in the attention projections, redundantly, every step.

The cache stores those keys and values once. Generating a token then costs one
new K/V pair per layer, and generation becomes O(n) in projection work.

The cost is memory, and it is not small.

## The memory arithmetic

For one token, across the whole model:

```
bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
                  ↑
                  keys and values
```

This is `ModelMetadata::kv_bytes_per_token()` — defined once, so nothing
recomputes it and drifts.

A concrete case. Llama-3-8B in FP16: 32 layers, 8 KV heads (grouped-query),
head dim 128.

```
2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes = 128 KiB per token
```

At 8192 tokens of context, **one sequence needs 1 GiB of KV cache.** On an 80 GB
A100 holding ~16 GB of weights, the cache budget is roughly 60 GB — about 60
maximum-length sequences, if memory were perfectly packed.

It is the packing that this design is about.

### What grouped-query attention buys

Note `num_kv_heads`, not `num_attention_heads`, in the formula. Llama-3-8B has
32 query heads but 8 KV heads. Multi-head attention would need:

```
2 × 32 × 32 × 128 × 2 = 512 KiB per token   — 4× more
```

GQA is a *cache* optimization as much as a compute one. `uses_gqa()` and
`gqa_group_size()` expose it because the difference determines how many
requests fit.

## The naive approach and why it fails

The obvious implementation gives each request one contiguous buffer sized to
its maximum possible length.

```text
Request A (max 2048, using 100)   Request B (max 2048, using 1900)
├────┬───────────────────────┤    ├──────────────────┬────────────┤
│used│      reserved waste   │    │       used       │   waste    │
└────┴───────────────────────┘    └──────────────────┴────────────┘
```

Two failures:

1. **Internal fragmentation.** Output length is unknown in advance, so the
   reservation must assume the maximum. A request that generates 100 tokens
   against a 2048 reservation wastes 95% of its allocation. Measured across a
   realistic workload, most reserved KV memory is never written.

2. **External fragmentation.** Contiguous buffers of varying size, allocated
   and freed in varying order, leave holes. Enough total memory can be free
   while no single hole is large enough for the next request.

## Paged blocks

The cache is divided into fixed-size **blocks**. A sequence gets a **block
table** mapping its logical positions to physical blocks, which need not be
adjacent or ordered.

```text
Physical pool
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │
└────┴────┴────┴────┴────┴────┴────┴────┘
   ▲         ▲    ▲              ▲
   │         │    │              │
   └─────────┴────┘              │
   Sequence A: [B0, B2, B3]      │
                                 │
   Sequence B: [B5] ─────────────┘
```

This is paged virtual memory, applied to attention state. It fixes both
problems:

- **No external fragmentation.** Every free block is interchangeable, so any
  free block satisfies any request for a block.
- **Internal fragmentation is bounded by one block per sequence.** A sequence
  wastes at most `block_size - 1` token slots in its final partial block,
  regardless of how long it eventually grows.

Growth is incremental: `append_token` allocates a block only when the current
final block fills, which for `block_size = 16` is once every 16 decode steps.

## Choosing the block size

The default is **16**. The tradeoff:

| | Small blocks (4) | Large blocks (128) |
|---|---|---|
| Waste per sequence | ≤ 3 tokens | ≤ 127 tokens |
| Block table entries per 2048 tokens | 512 | 16 |
| Allocations per 2048 decode steps | 512 | 16 |
| Prefix sharing granularity | fine | coarse |
| Kernel gather efficiency | poor | good |

**Why not smaller.** Below about 4, the block table becomes comparable in size
to the data it indexes, and the attention kernel's gather over scattered blocks
loses coalescing — each block boundary is a potential non-contiguous memory
access.

**Why not larger.** Internal waste is `block_size - 1` tokens per sequence in
the worst case. At 256 sequences and `block_size = 128`, that is up to 32,768
wasted token slots — 4 GiB for the Llama-3-8B numbers above. Large blocks also
make prefix sharing coarse: two prompts sharing 100 tokens share nothing at all
if the block size is 128.

**Why a power of two.** Position-to-block arithmetic becomes a shift and a
mask rather than a division. `CacheConfig::validate` enforces it, and the
allowed range is 4–256.

16 sits where waste (≤15 tokens/sequence, negligible) meets acceptable
metadata and reasonable sharing granularity. This is a starting point, not a
measured optimum — the value should be revisited against a real workload once
end-to-end benchmarking exists, and `docs/performance-journal.md` is where such
a measurement will be recorded.

## Reference counting and sharing

Each block carries a `ref_count`. A block is reclaimable exactly when it
reaches zero. Two sequences with a common prompt prefix can point at the same
physical blocks, and neither's completion frees memory the other still needs.

Freed blocks go to the **back** of a FIFO free list rather than being reused at
once. This is what makes prefix caching work without a separate cache tier: a
block whose refcount drops to zero keeps its contents and its hash, so a later
request with the same prefix can reclaim it by hash. Under pressure, the oldest
such block is recycled first — LRU eviction falling out of the FIFO ordering
for free.

## Prefix caching

Reusing a block's KV values is sound only if recomputing them would produce the
same values. Attention output depends on all preceding context, so it is not
enough for two blocks to hold the same tokens — their entire histories must
match.

That is encoded by **chaining** the hash:

```text
hash(block_0) = H(∅,            tokens[0..16])
hash(block_1) = H(hash(block_0), tokens[16..32])
hash(block_2) = H(hash(block_1), tokens[32..48])
```

Block *n* matches only if blocks 0..=n all match. `hash_block` also mixes in
the token count, so a short final block cannot collide with a longer one
sharing its leading tokens.

### Three correctness guards

1. **Token verification on lookup.** The index stores each cached block's
   tokens and compares them on every hit. A hash collision degrades to a miss,
   never to corrupt KV data. The hash is not cryptographic and does not need to
   be.

2. **Only full blocks are published.** A partial block will still be written
   to; sharing it would let one sequence observe another's tokens appearing
   underneath it.

3. **Publication happens after compute, not after allocation.**
   `commit_prefill` runs once the forward pass has actually written the KV
   entries. Publishing at allocation time would let another sequence adopt
   blocks holding uninitialized memory.

### The stale-entry hazard

The index holds no references of its own — deliberately, so the cache cannot
pin memory and starve live requests. That means a cached block may be recycled
while an index entry still names it.

`BlockPool::allocate` therefore returns the hash the recycled block *used* to
carry, and the manager drops that entry before anything can look it up. Without
this, a later lookup could adopt a block whose contents had been overwritten by
an unrelated sequence — a silent correctness failure producing plausible but
wrong output. The regression test is
`recycling_a_cached_block_invalidates_its_index_entry`.

## Out-of-memory behaviour

Exhaustion is a normal operating condition, not an exceptional one. A busy
server runs near capacity by design.

- `allocate` is **failure-atomic**: if the pool runs out partway through, every
  block taken by that call is released before returning. A partially-allocated
  sequence would leak blocks that nothing owns, invisibly, until the pool was
  exhausted.
- `EngineError::CacheExhausted` is classified `is_retryable()`, so the API
  layer returns 503 with `Retry-After` rather than a terminal error.
- The scheduler responds by preempting the newest running sequence rather than
  failing the request. See [scheduler.md](scheduler.md).

## Concurrency

`KvCacheManager` is **not** internally synchronized and is mutated through
`&mut self`. It is owned by the single engine step loop.

This is deliberate. Wrapping it in a mutex would invite callers to interleave
allocation decisions with scheduling decisions, which is precisely the race
that produces double-allocation bugs in a block allocator. Serializing the
bookkeeping costs nothing measurable — the engine is bound on the forward pass,
not on block arithmetic — and removes the entire bug class.

## Status

| Feature | State |
|---|---|
| Block pool with reference counting | implemented |
| Per-sequence block tables | implemented |
| Failure-atomic allocation | implemented |
| Chained-hash prefix caching | implemented |
| LRU reclamation of cached blocks | implemented |
| Copy-on-write for forked sequences | planned |
| CPU swap tier for preemption | planned |
| Sliding-window / attention-sink eviction | planned |

Memory sizing from actual device capacity requires a GPU backend and is
`planned`; `CacheConfig::num_blocks` must currently be set explicitly.
