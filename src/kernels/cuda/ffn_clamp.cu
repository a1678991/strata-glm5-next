// src/kernels/cuda/ffn_clamp.cu - see include/strata/kernels/ffn_clamp.hpp.
//
// The fused SwiGLU + Q8_1 kernel follows native_mmvq.cu's native_quantize_q8_1_kernel block for block (one 32-lane
// warp per Q8_1 block, the amax / sum warp reductions, q8_1_finite / q8_1_quant / q8_1_ds), so its blocks are the ones
// native_quantize_q8_1 would make of the same h.  h itself is ggml_swiglu_clamp's element with a full-precision expf.
#include "strata/kernels/ffn_clamp.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "Q8_1 block");

constexpr int THREADS = 256;

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}

__global__ void __launch_bounds__(THREADS) swiglu_clamp_q8_1_kernel(const float* __restrict__ gate,
                                                                    const float* __restrict__ up, float limit,
                                                                    Q81* __restrict__ y, int n) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
#endif
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;   // n is a multiple of 32: only whole warps return
    float g = gate[i], u = up[i];
    if (limit > 0.0f) {   // std::min(g, limit), std::clamp(u, -limit, limit) (ops.cpp 3614-3615)
        g = limit < g ? limit : g;
        u = u < -limit ? -limit : (limit < u ? limit : u);
    }
#if defined(__HIPCC__)
    const float xi = (g / (1.0f + expf(-g))) * u;
#else
    const float xi = __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), u);
#endif
    const float amax = wmax(fabsf(xi));
    const float sum = wsum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / 32].qs[i % 32] = q;
    if (i % 32 == 0) y[i / 32].ds = q8_1_ds(d, sum);
}

size_t align256(size_t b) { return (b + 255) & ~(size_t) 255; }

}  // namespace

void native_swiglu_clamp_quantize_q8_1(const float* gate, const float* up, float limit, void* h_q8_1, int n_ff,
                                       int ncols, void* stream) {
    if (!gate || !up || !h_q8_1 || n_ff <= 0 || n_ff % 32 != 0 || ncols < 1 || ncols > 8)
        throw std::invalid_argument("native_swiglu_clamp_quantize_q8_1: null buffer, n_ff not a multiple of 32, or "
                                    "ncols outside 1..8");
    const int n = n_ff * ncols;
    swiglu_clamp_q8_1_kernel<<<(unsigned) ((n + THREADS - 1) / THREADS), THREADS, 0, (cudaStream_t) stream>>>(
        gate, up, limit, static_cast<Q81*>(h_q8_1), n);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native_swiglu_clamp_quantize_q8_1: ") + cudaGetErrorString(e));
}

std::size_t ffn_clamp_scratch_bytes(int n_embd, int n_ff, int ncols) {
    return align256(native_q8_1_bytes(n_embd, ncols)) + 2 * align256((size_t) ncols * n_ff * sizeof(float)) +
           align256(native_q8_1_bytes(n_ff, ncols));
}
const float* ffn_clamp_gate(const void* scratch, int n_embd, int, int ncols) {
    return (const float*) ((const uint8_t*) scratch + align256(native_q8_1_bytes(n_embd, ncols)));
}
const float* ffn_clamp_up(const void* scratch, int n_embd, int n_ff, int ncols) {
    return (const float*) ((const uint8_t*) scratch + align256(native_q8_1_bytes(n_embd, ncols)) +
                           align256((size_t) ncols * n_ff * sizeof(float)));
}
const void* ffn_clamp_h_q8_1(const void* scratch, int n_embd, int n_ff, int ncols) {
    return (const uint8_t*) scratch + align256(native_q8_1_bytes(n_embd, ncols)) +
           2 * align256((size_t) ncols * n_ff * sizeof(float));
}

void ffn_clamp(const FfnClampWeights& w, const float* x, float* out, int n_embd, int n_ff, int ncols, float limit,
               void* scratch, void* stream) {
    if (!w.gate || !w.up || !w.down || !x || !out || !scratch || !native_mmvq_supported(w.gate_type) ||
        !native_mmvq_supported(w.up_type) || !native_mmvq_supported(w.down_type))
        throw std::invalid_argument("ffn_clamp: null buffer or a weight type native_mmvq does not take");
    uint8_t* xq = (uint8_t*) scratch;
    float* gate = const_cast<float*>(ffn_clamp_gate(scratch, n_embd, n_ff, ncols));
    float* up = const_cast<float*>(ffn_clamp_up(scratch, n_embd, n_ff, ncols));
    void* hq = const_cast<void*>(ffn_clamp_h_q8_1(scratch, n_embd, n_ff, ncols));
    native_quantize_q8_1(x, xq, n_embd, ncols, stream);
    native_mmvq(w.gate_type, w.gate, xq, gate, n_embd, n_ff, ncols, stream);
    native_mmvq(w.up_type, w.up, xq, up, n_embd, n_ff, ncols, stream);
    native_swiglu_clamp_quantize_q8_1(gate, up, limit, hq, n_ff, ncols, stream);
    native_mmvq(w.down_type, w.down, hq, out, n_ff, n_embd, ncols, stream);
}

}  // namespace strata::kernels
