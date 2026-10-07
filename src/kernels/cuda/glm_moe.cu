// src/kernels/cuda/glm_moe.cu - see include/strata/kernels/glm_moe.hpp.
#include "strata/kernels/glm_moe.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// CPU: a row whose cpu_map entry is >= 0 is read from cpu_rows instead of parts.  The arithmetic and its order are the
// same in both instantiations; only where a row's floats come from differs.
template <bool CPU>
__global__ void glm_moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                                       const float* __restrict__ shared, float* __restrict__ out, int n_embd, int k,
                                       const float* __restrict__ cpu_rows, const int32_t* __restrict__ cpu_map) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    auto row = [&](int j) -> const float* {
        if constexpr (CPU) {
            const int32_t c = cpu_map[t * k + j];
            if (c >= 0) return cpu_rows + (size_t) c * n_embd + i;
        }
        return parts + ((size_t) t * k + j) * n_embd + i;
    };
    const float* wt = w + (size_t) t * k;
    float acc = __fmul_rn(*row(0), wt[0]);
    for (int j = 1; j < k; ++j) acc = __fadd_rn(acc, __fmul_rn(*row(j), wt[j]));
    out[(size_t) t * n_embd + i] = __fadd_rn(acc, shared[(size_t) t * n_embd + i]);
}

}  // namespace

void glm_moe_combine(const float* parts, const float* w, const float* shared, float* out, int n_embd, int k, int n_tok,
                     void* stream, const float* cpu_rows, const int32_t* cpu_map) {
    if (n_tok <= 0 || n_embd <= 0 || k <= 0) return;
    const dim3 grid((unsigned) ((n_embd + 255) / 256), (unsigned) n_tok);
    if (cpu_rows != nullptr && cpu_map != nullptr)
        glm_moe_combine_kernel<true><<<grid, 256, 0, (cudaStream_t) stream>>>(parts, w, shared, out, n_embd, k, cpu_rows,
                                                                               cpu_map);
    else
        glm_moe_combine_kernel<false><<<grid, 256, 0, (cudaStream_t) stream>>>(parts, w, shared, out, n_embd, k,
                                                                                nullptr, nullptr);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::fprintf(stderr, "glm_moe_combine: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
