// RMSNorm kernel.
//
// STATUS: WRITTEN BUT NEVER COMPILED OR RUN.
// No CUDA toolkit or NVIDIA GPU was available to this project. This code has
// not been through nvcc, has not been tested for correctness against the CPU
// reference, and has not been benchmarked. Treat every claim below as a design
// intention, not a measured result.
//
// ---------------------------------------------------------------------------
//
// RMSNorm normalizes each row by its root-mean-square and scales by a learned
// per-column gain:
//
//     y[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]
//
// It differs from LayerNorm by omitting mean subtraction and bias, which saves
// a pass over the data. See crates/orion-models/src/tensor.rs for the reference
// implementation this must match.
//
// Parallelization: one block per row, one thread per element (grid-strided when
// hidden_size exceeds the block). The sum of squares is a block-wide reduction,
// which is the only synchronization point.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace orion {

// Warp size is 32 on every NVIDIA architecture to date. Hard-coding it lets the
// shuffle reduction below assume a fixed lane count.
constexpr int kWarpSize = 32;

// Reduces a value across one warp using shuffle instructions.
//
// Shuffles rather than shared memory: they avoid a round trip through memory
// and need no __syncthreads(), because a warp executes in lockstep.
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Reduces across a whole block: warps reduce internally, then the per-warp
// partials are reduced by the first warp.
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    // Only the first warp participates in the final reduction. Threads beyond
    // the number of active warps contribute zero.
    const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// RMSNorm over a [num_rows, hidden_size] tensor, in place.
//
// One block per row. `eps` is added inside the square root, matching the
// reference implementation exactly -- adding it outside would change the result
// in the small-magnitude case that eps exists to protect.
//
// Accumulation is in float even when the data is half, because summing 8192
// squared values in half precision loses enough mantissa to shift the
// normalization visibly. The CPU reference accumulates in f64 for the same
// reason; float is the practical compromise on device.
extern "C" __global__ void orion_rmsnorm_f32(
    float* __restrict__ x,           // [num_rows, hidden_size], modified in place
    const float* __restrict__ gamma, // [hidden_size]
    const int hidden_size,
    const float eps) {

    extern __shared__ float shared[];

    const int row = blockIdx.x;
    float* row_data = x + (size_t)row * hidden_size;

    // Each thread accumulates the squares of its strided slice.
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = row_data[i];
        local_sum += v * v;
    }

    float total = block_reduce_sum(local_sum, shared);

    // Thread 0 computes the scale and publishes it; every thread reads it back.
    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = rsqrtf(total / (float)hidden_size + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_data[i] = row_data[i] * scale * gamma[i];
    }
}

// Half-precision variant. Reads and writes __half, accumulates in float.
extern "C" __global__ void orion_rmsnorm_f16(
    __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const int hidden_size,
    const float eps) {

    extern __shared__ float shared[];

    const int row = blockIdx.x;
    __half* row_data = x + (size_t)row * hidden_size;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = __half2float(row_data[i]);
        local_sum += v * v;
    }

    float total = block_reduce_sum(local_sum, shared);

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = rsqrtf(total / (float)hidden_size + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float v = __half2float(row_data[i]) * scale * __half2float(gamma[i]);
        row_data[i] = __float2half(v);
    }
}

}  // namespace orion
