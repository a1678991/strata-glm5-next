// include/strata/kernels/ffn_clamp.hpp - GLM-5.3-Flash's dense SwiGLU FFNs with the clamp: the three leading dense
// layers (ffn_{gate,up,down} [4096, 12288] Q8_0) and every MoE layer's shared expert (ffn_*_shexp [4096, 2048] Q8_0,
// added plain - no scalar gate, unlike Qwen's shared_expert()).
//
//   out = W_down . (silu(min(W_gate x, limit)) * clamp(W_up x, -limit, limit))
//
// llama.cpp f1cee99 build_ffn (src/llama-graph.cpp 1850-1857): limit = swiglu_clamp_shexp[il] (10.0 in the GGUF for
// every layer, the dense ones included) and ggml_swiglu_clamp (ggml-cpu ops.cpp 3614-3616).  The projections are
// Strata's native GEMV (native_mmvq.hpp: any type it takes, Q8_0 here) on one shared Q8_1 quantization of x; the
// clamped SwiGLU and the Q8_1 quantization of its output for the down GEMV are one kernel.  ncols 1..8 tokens.
// Device memory, the caller's stream, no allocation, no synchronization.
#pragma once

#include <cstddef>

namespace strata::kernels {

/// h = silu(min(gate, limit)) * clamp(up, -limit, limit), written as Q8_1 blocks in native_mmvq's layout (36 bytes per
/// 32 values, column j at block j * n_ff / 32), exactly what native_quantize_q8_1 makes of that h.  limit <= 0: the
/// plain SwiGLU.  silu with full-precision expf, as ggml-cpu.  n_ff a multiple of 32.
void native_swiglu_clamp_quantize_q8_1(const float* gate, const float* up, float limit, void* h_q8_1, int n_ff,
                                       int ncols, void* stream);

struct FfnClampWeights {
    int gate_type = 8, up_type = 8, down_type = 8;   ///< ggml type ids (native_mmvq_supported); GLM: Q8_0 (8)
    const void* gate = nullptr;                      ///< n_ff rows of n_embd
    const void* up = nullptr;                        ///< n_ff rows of n_embd
    const void* down = nullptr;                      ///< n_embd rows of n_ff
};

/// Scratch for ffn_clamp: x's Q8_1, gate and up (F32, ncols x n_ff each) and h's Q8_1.
std::size_t ffn_clamp_scratch_bytes(int n_embd, int n_ff, int ncols);
/// out (ncols x n_embd) = the clamped SwiGLU FFN of x (ncols x n_embd).  gate/up F32 stay in the scratch:
/// ffn_clamp_gate(scratch) / ffn_clamp_up(scratch, ...) point at them (tests and the integration's debugging).
void ffn_clamp(const FfnClampWeights& w, const float* x, float* out, int n_embd, int n_ff, int ncols, float limit,
               void* scratch, void* stream);
const float* ffn_clamp_gate(const void* scratch, int n_embd, int n_ff, int ncols);
const float* ffn_clamp_up(const void* scratch, int n_embd, int n_ff, int ncols);
const void* ffn_clamp_h_q8_1(const void* scratch, int n_embd, int n_ff, int ncols);

}  // namespace strata::kernels
