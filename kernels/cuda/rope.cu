// Rotary Position Embedding (RoPE) kernel.
//
// STATUS: WRITTEN BUT NEVER COMPILED OR RUN.
// No CUDA toolkit or NVIDIA GPU was available. Not compiled, not tested for
// correctness against the CPU reference, not benchmarked.
//
// ---------------------------------------------------------------------------
//
// RoPE rotates pairs of dimensions within each attention head by an angle that
// depends on absolute position, so that the dot product between two rotated
// vectors depends only on their *relative* position.
//
// The pairing convention is load-bearing: dimension i pairs with i + head_dim/2,
// NOT with i + 1. Hugging Face checkpoints are trained with this "rotate-half"
// layout, and getting it wrong yields a model that still produces fluent text
// while attending to the wrong positions -- a failure no shape check catches.
// See crates/orion-models/src/tensor.rs, which has a test pinning exactly this.
//
// Parallelization: one thread per (token, head, dimension-pair). Each thread
// does two loads, two stores and four multiply-adds, with no communication.
// This is purely memory bound.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace orion {

// Applies RoPE to a [num_tokens, num_heads, head_dim] tensor, in place.
//
// `cos_table` and `sin_table` are [max_positions, head_dim/2], precomputed on
// the host at load time -- they depend only on position and head dimension,
// never on the data, so recomputing them per token would be pure waste.
//
// `positions` gives the absolute position of each token, which under continuous
// batching is NOT simply the row index: a decode step submits token N of one
// sequence alongside token M of another.
extern "C" __global__ void orion_rope_f32(
    float* __restrict__ x,            // [num_tokens, num_heads, head_dim]
    const float* __restrict__ cos_table,  // [max_positions, head_dim/2]
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,    // [num_tokens]
    const int num_tokens,
    const int num_heads,
    const int head_dim) {

    const int half_dim = head_dim / 2;
    const int total = num_tokens * num_heads * half_dim;

    // Grid-stride loop so one launch configuration covers any batch size.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {

        const int pair = idx % half_dim;
        const int head = (idx / half_dim) % num_heads;
        const int token = idx / (half_dim * num_heads);

        const int pos = positions[token];
        const float c = cos_table[(size_t)pos * half_dim + pair];
        const float s = sin_table[(size_t)pos * half_dim + pair];

        float* head_base = x + ((size_t)token * num_heads + head) * head_dim;

        // The rotate-half pairing: i with i + half_dim.
        const float x0 = head_base[pair];
        const float x1 = head_base[pair + half_dim];

        head_base[pair] = x0 * c - x1 * s;
        head_base[pair + half_dim] = x0 * s + x1 * c;
    }
}

// Half-precision variant. The rotation itself is computed in float: the angles
// come from a float table, and rounding the intermediate products to half would
// accumulate error across many layers.
extern "C" __global__ void orion_rope_f16(
    __half* __restrict__ x,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    const int num_tokens,
    const int num_heads,
    const int head_dim) {

    const int half_dim = head_dim / 2;
    const int total = num_tokens * num_heads * half_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {

        const int pair = idx % half_dim;
        const int head = (idx / half_dim) % num_heads;
        const int token = idx / (half_dim * num_heads);

        const int pos = positions[token];
        const float c = cos_table[(size_t)pos * half_dim + pair];
        const float s = sin_table[(size_t)pos * half_dim + pair];

        __half* head_base = x + ((size_t)token * num_heads + head) * head_dim;

        const float x0 = __half2float(head_base[pair]);
        const float x1 = __half2float(head_base[pair + half_dim]);

        head_base[pair] = __float2half(x0 * c - x1 * s);
        head_base[pair + half_dim] = __float2half(x0 * s + x1 * c);
    }
}

}  // namespace orion
