# CUDA

## Status

**No CUDA code in this repository has been compiled or executed.**

The development machine has no NVIDIA GPU and no CUDA toolkit — `nvcc` and
`nvidia-smi` are both absent. Three kernels are written in `kernels/cuda/`, and
the Rust integration surface exists in `crates/orion-cuda`, but nothing has been
through a compiler that understands `.cu` files, nothing has been checked for
numerical correctness, and nothing has been benchmarked.

This document describes the design and the validation process that will gate it.
It makes no performance claim.

## Why kernels at all

Three operations in a decoder layer are memory bound and poorly served by
generic library calls:

**RMSNorm** — a reduction followed by a scaled write. A library implementation
reads the row twice: once to reduce, once to scale. Fusing them halves the
traffic.

**RoPE** — two loads, two stores, four multiply-adds per element pair, with
table lookups. There is no BLAS call for this; it is either a custom kernel or
several tensor-library operations with intermediates between them.

**SwiGLU** — `SiLU(gate) * up`. Unfused, it reads and writes an intermediate
that is four times the hidden size, the largest tensor in the layer. Fused, it
reads two inputs and writes one output: the minimum possible traffic.

Matrix multiplication is deliberately **not** on this list. cuBLAS is written by
people with access to hardware documentation nobody outside NVIDIA has, and
beating it is not a reasonable goal for this project.

## Kernel designs

### RMSNorm (`kernels/cuda/rmsnorm.cu`)

One block per row, grid-strided within the row. The sum of squares is a
block-wide reduction using warp shuffles, then a shared-memory reduction across
warps.

Shuffles rather than shared memory for the intra-warp step: they avoid a memory
round trip and need no `__syncthreads()`, because a warp executes in lockstep.

Accumulation is in `float` even for half-precision data. Summing 8192 squared
values in `f16` loses enough mantissa to shift the normalization visibly — the
CPU reference accumulates in `f64` for the same reason.

### RoPE (`kernels/cuda/rope.cu`)

One thread per `(token, head, dimension-pair)`. No communication, no shared
memory; purely memory bound.

The pairing convention is the part that matters: dimension `i` pairs with
`i + head_dim/2`, **not** `i + 1`. Hugging Face checkpoints are trained with
this "rotate-half" layout. Getting it wrong yields a model that produces fluent
text while attending to the wrong positions, which no shape check catches. The
CPU reference has a test pinning exactly this, and the CUDA version must match.

Positions come from a per-token array rather than being derived from the row
index, because under continuous batching a step contains token 500 of one
sequence beside token 3 of another.

### SwiGLU (`kernels/cuda/swiglu.cu`)

Elementwise, grid-strided, writing into the gate buffer in place so the caller
needs no extra allocation.

The half-precision variant processes `__half2` pairs, halving the number of
memory transactions — which matters for a kernel that is entirely memory bound.
The activation is computed in `float` and only the result narrowed.

`__expf` rather than `expf`: its roughly 2 ulp accuracy is far tighter than the
precision the activation is consumed at, and correctly-rounded `expf` is much
slower.

## The validation gate

Every kernel passes all five steps, in order, before any performance claim:

**1. It compiles.** `nvcc` accepts it for a real target architecture.

**2. It is numerically correct.** Output matches the CPU reference in
`orion-models::tensor` within tolerance, over shapes covering the edge cases:
single row, widths that are not multiples of the warp size, extreme values,
all-zero inputs.

Tolerances live in `orion-cuda::Tolerance` and are **not** bit-exactness.
A GPU kernel will not match bit-for-bit, and demanding it would be wrong:
reduction order differs, fused multiply-add changes rounding, and fast-math
intrinsics trade accuracy for speed on purpose. `F32` allows 1e-5; `F16` allows
1e-2, because half precision has roughly eleven bits of mantissa and holding it
to `f32` tolerance would fail correct code.

The comparison harness (`orion_cuda::validate`) is **implemented and tested
now**, without a GPU. Writing the checker after the kernel is how a subtly wrong
kernel gets declared correct.

**3. End-to-end output is unchanged.** A full forward pass on GPU produces the
same logits as the CPU path, within tolerance, for the same input. Per-kernel
correctness does not compose automatically.

**4. It is measured.** Benchmarked against the CPU reference *and* against an
unfused sequence of library calls, because "faster than our own slow CPU code"
is not a meaningful claim.

`Backend::synchronize` is called around every measurement. CUDA launches are
asynchronous: without synchronization a kernel appears to take microseconds
regardless of what it does. This is the single easiest way to publish a false
GPU benchmark, and the trait method exists specifically to prevent it.

**5. It is recorded.** Numbers go in `docs/performance-journal.md` with the GPU
model, driver version, CUDA version, dtype and shapes.

## Building, when hardware exists

```bash
# The cuda feature is off by default; the workspace builds and tests without it.
cargo build --release --features cuda
```

`crates/orion-cuda/build.rs` will invoke `nvcc` and link the result. That build
script does not exist yet, because there is nothing to test it against.

## Profiling

**Nsight Systems** for the timeline — where time goes across kernels, memory
copies and CPU gaps:

```bash
nsys profile --trace=cuda,nvtx ./target/release/orion serve --model ...
```

**Nsight Compute** for one kernel's occupancy, memory throughput and achieved
bandwidth:

```bash
ncu --set full -k orion_rmsnorm_f32 ./target/release/orion ...
```

For a memory-bound kernel the number that matters is achieved bandwidth as a
fraction of peak. Anything below roughly 70% means the access pattern is wrong,
and no amount of instruction-level tuning will fix that.

## What is not designed here

**Paged attention on GPU** is the hard one, and is not attempted above. The CPU
implementation gathers K and V through a block table one position at a time,
which is correct but would be catastrophic on a GPU. A real implementation needs
the FlashAttention approach — tiling the computation so the attention matrix
never materializes — adapted to gather through a block table. That is
substantially harder than the three kernels above and needs hardware to develop
against.

**Multi-stream execution**, **CUDA graphs** to amortize launch overhead across a
32-layer model, and a **device memory pool** are all absent from the `Backend`
trait. Each would need it widened, which ADR 005 notes as an accepted future
cost.
