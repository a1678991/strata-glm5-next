// src/kernels/cuda/glm_moe.cu - see include/strata/kernels/glm_moe.hpp.
#include "strata/kernels/glm_moe.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

__global__ void glm_moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                                       const float* __restrict__ shared, float* __restrict__ out, int n_embd, int k) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float* p = parts + (size_t) t * k * n_embd + i;
    const float* wt = w + (size_t) t * k;
    float acc = __fmul_rn(p[0], wt[0]);
    for (int j = 1; j < k; ++j) acc = __fadd_rn(acc, __fmul_rn(p[(size_t) j * n_embd], wt[j]));
    out[(size_t) t * n_embd + i] = __fadd_rn(acc, shared[(size_t) t * n_embd + i]);
}

}  // namespace

void glm_moe_combine(const float* parts, const float* w, const float* shared, float* out, int n_embd, int k, int n_tok,
                     void* stream) {
    if (n_tok <= 0 || n_embd <= 0 || k <= 0) return;
    const dim3 grid((unsigned) ((n_embd + 255) / 256), (unsigned) n_tok);
    glm_moe_combine_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(parts, w, shared, out, n_embd, k);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::fprintf(stderr, "glm_moe_combine: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
