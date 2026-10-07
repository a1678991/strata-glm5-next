// include/strata/kernels/cpu/native_expert_clamp.hpp - GLM-5.3-Flash's clamped SwiGLU for the CPU pool's native
// expert rows.  A separate header so that native_expert.hpp (the pool's interface) is not changed by this port.
//
//   ff[t][r] = silu(min(g, limit)) * clamp(u, -limit, limit)     (ggml_swiglu_clamp, ggml-cpu ops.cpp 3614-3616 at
//                                                                 llama.cpp f1cee99; GLM's swiglu_clamp_exp is 10)
//
// limit <= 0 is native_gu_rows exactly: native_gu_rows is native_gu_rows_clamp(..., 0.0f), and every kernel it
// reaches (ggml-cpu's vec_dot loop, iq512/iq256/kq256_gu_rows) runs its original SwiGLU loop when the limit is 0.
#pragma once

#include "strata/kernels/cpu/native_expert.hpp"

#include <cmath>

namespace strata::kernels::cpu {

/// ggml_compute_forward_swiglu_clamp_f32's element: std::min(g, limit) and std::clamp(u, -limit, limit) written out as
/// those functions' comparisons (the same result, NaN included), then gate / (1 + exp(-gate)) * up.  `static`: this
/// header is included by translation units compiled for different ISAs (iq_avx512.cpp, iq_avx2.cpp, ...), and an
/// inline function with external linkage could be merged into one AVX-512 copy for all of them.
static inline float swiglu_clamped(float g, float u, float limit) {
    g = limit < g ? limit : g;
    u = u < -limit ? -limit : (limit < u ? limit : u);
    return (g / (1.f + std::exp(-g))) * u;
}

/// native_gu_rows with the clamp (limit > 0) or without it (limit <= 0: bit for bit native_gu_rows).
void native_gu_rows_clamp(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                          int r0, int r1, float limit);

}  // namespace strata::kernels::cpu
