// Fused SwiGLU activation kernel.
//
// STATUS: WRITTEN BUT NEVER COMPILED OR RUN.
// No CUDA toolkit or NVIDIA GPU was available. Not compiled, not tested for
// correctness against the CPU reference, not benchmarked.
//
// ---------------------------------------------------------------------------
//
// SwiGLU computes, elementwise:
//
//     out[i] = SiLU(gate[i]) * up[i]
//            = gate[i] * sigmoid(gate[i]) * up[i]
//
// Fusing this is the point. Unfused, it costs three passes over an
// intermediate that is 4x the hidden size -- the largest tensor in the layer.
// One kernel reads gate and up once each and writes once, which is the minimum
// possible traffic for the operation.
//
// This is memory bound, not compute bound: two loads and one store per element
// against a handful of arithmetic operations. The expected win over an unfused
// sequence is therefore roughly the ratio of memory traffic, not of FLOPs.
// Whether it materializes is an empirical question that has not been answered
// here, because no GPU was available to answer it on.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace orion {

__device__ __forceinline__ float silu(float x) {
    // __expf is the fast-math intrinsic. Its accuracy (~2 ulp) is far tighter
    // than the precision the activation is consumed at, and it avoids the much
    // slower correctly-rounded expf.
    return x / (1.0f + __expf(-x));
}

// out = SiLU(gate) * up, elementwise over `n` elements.
//
// Writes into `gate` in place, matching the CPU reference's signature, so the
// caller needs no extra allocation for the intermediate.
extern "C" __global__ void orion_swiglu_f32(
    float* __restrict__ gate,      // [n], modified in place
    const float* __restrict__ up,  // [n]
    const int n) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        gate[i] = silu(gate[i]) * up[i];
    }
}

// Half-precision variant using __half2 for vectorized loads.
//
// Processing two elements per thread halves the number of memory transactions,
// which matters for a kernel that is entirely memory bound. `n` must be even;
// the caller pads the intermediate dimension, which is already a multiple of
// 128 in every architecture this supports.
extern "C" __global__ void orion_swiglu_f16(
    __half2* __restrict__ gate,
    const __half2* __restrict__ up,
    const int n_half2) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_half2;
         i += gridDim.x * blockDim.x) {

        const float2 g = __half22float2(gate[i]);
        const float2 u = __half22float2(up[i]);

        // Activation computed in float; only the result is narrowed.
        const float2 r = make_float2(silu(g.x) * u.x, silu(g.y) * u.y);
        gate[i] = __float22half2_rn(r);
    }
}

}  // namespace orion
