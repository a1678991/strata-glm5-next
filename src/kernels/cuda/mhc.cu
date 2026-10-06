// src/kernels/cuda/mhc.cu - GLM-5.3-Flash mHC mixing kernels; see include/strata/kernels/mhc.hpp.
//
// Arithmetic follows ggml-cpu's fused ops (ops.cpp 11337-11645) and the graph around them in G:454-547 operation by
// operation: separate ggml ops are separate roundings here too (a product is never contracted into the following
// add when ggml computes them in two ops), so the products below go through mul_rn.  Full-precision expf / division.
#include "strata/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HC = 4;
constexpr int MIX = (2 + HC) * HC;   // 24

// a product rounded on its own (not fused into a following add)
__device__ __forceinline__ float mul_rn(float a, float b) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
    return a * b;
#else
    return __fmul_rn(a, b);
#endif
}
__device__ __forceinline__ float add_rn(float a, float b) {
#if defined(__HIPCC__)
    return a + b;
#else
    return __fadd_rn(a, b);
#endif
}
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}

constexpr int NORM_THREADS = 1024;

__global__ void __launch_bounds__(NORM_THREADS) mhc_flat_norm_kernel(const float* __restrict__ x,
                                                                     float* __restrict__ flat, int n, float eps) {
    __shared__ float part[NORM_THREADS / 32];
    const float* xt = x + (size_t) blockIdx.x * n;
    float* ft = flat + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += NORM_THREADS) ss = fmaf(xt[i], xt[i], ss);
    ss = warp_sum32(ss);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = part[threadIdx.x];
        v = warp_sum32(v);
        if (threadIdx.x == 0) part[0] = v;
    }
    __syncthreads();
    // ops.cpp rms_norm: mean = sum / ne00; scale = 1.0f / sqrtf(mean + eps); y = x * scale
    const float mean = part[0] / (float) n;
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = threadIdx.x; i < n; i += NORM_THREADS) ft[i] = xt[i] * scale;
}

// ggml_dsv4_hc_comb_norm_cols / _rows (ops.cpp 11339-11369): comb[idst + hc * isrc]
__device__ void norm_cols(float* c, float eps) {
#pragma unroll
    for (int d = 0; d < HC; ++d) {
        float sum = eps;
#pragma unroll
        for (int s = 0; s < HC; ++s) sum += c[d + HC * s];
        const float inv = 1.0f / sum;
#pragma unroll
        for (int s = 0; s < HC; ++s) c[d + HC * s] *= inv;
    }
}
__device__ void norm_rows(float* c, float eps) {
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        float sum = eps;
#pragma unroll
        for (int d = 0; d < HC; ++d) sum += c[d + HC * s];
        const float inv = 1.0f / sum;
#pragma unroll
        for (int d = 0; d < HC; ++d) c[d + HC * s] *= inv;
    }
}

constexpr int PRE_THREADS = 256;

// grid (ceil(n_embd / 256), T): every block derives the token's 24 mixing values (a few hundred flops) and writes
// its 256 values of cur; block x == 0 also writes post and comb.
__global__ void __launch_bounds__(PRE_THREADS) mhc_pre_kernel(const float* __restrict__ x,
                                                              const float* __restrict__ mixes,
                                                              const float* __restrict__ scale,
                                                              const float* __restrict__ base, float* __restrict__ cur,
                                                              float* __restrict__ post, float* __restrict__ comb,
                                                              int n_embd, float eps, int n_iter) {
    __shared__ float s_pre[HC];
    const int t = blockIdx.y;
    const float* m = mixes + (size_t) t * MIX;
    if (threadIdx.x < HC) {
        // G:476-480: view, ggml_mul(scale[0]), ggml_add(base), ggml_sigmoid, ggml_scale_bias(1, eps)
        const int h = threadIdx.x;
        s_pre[h] = sigmoid(add_rn(mul_rn(m[h], scale[0]), base[h])) + eps;
    } else if (threadIdx.x < 2 * HC && blockIdx.x == 0) {
        // G:482-486: ggml_mul(scale[1]), ggml_add(base), ggml_sigmoid, ggml_scale(2)
        const int h = threadIdx.x - HC;
        post[(size_t) t * HC + h] = 2.0f * sigmoid(add_rn(mul_rn(m[HC + h], scale[1]), base[HC + h]));
    } else if (threadIdx.x == 32 && blockIdx.x == 0) {
        // ggml_compute_forward_dsv4_hc_comb_f32 (ops.cpp 11371-11456) for one token
        float c[HC * HC];
        const float sc = scale[2];
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            float mx = -INFINITY;
#pragma unroll
            for (int d = 0; d < HC; ++d) {
                const int idx = d + HC * s;
                const float v = add_rn(mul_rn(m[2 * HC + idx], sc), base[2 * HC + idx]);
                c[idx] = v;
                mx = fmaxf(mx, v);
            }
            float sum = 0.0f;
#pragma unroll
            for (int d = 0; d < HC; ++d) {
                const int idx = d + HC * s;
                const float v = expf(c[idx] - mx);
                c[idx] = v;
                sum += v;
            }
            const float inv = 1.0f / sum;
#pragma unroll
            for (int d = 0; d < HC; ++d) {
                const int idx = d + HC * s;
                c[idx] = add_rn(mul_rn(c[idx], inv), eps);
            }
        }
        norm_cols(c, eps);
        for (int i = 1; i < n_iter; ++i) {
            norm_rows(c, eps);
            norm_cols(c, eps);
        }
#pragma unroll
        for (int i = 0; i < HC * HC; ++i) comb[(size_t) t * HC * HC + i] = c[i];
    }
    __syncthreads();
    // ggml_compute_forward_dsv4_hc_pre_f32 (ops.cpp 11477-11536), unscaled: sum += x * w over the streams
    const int i = blockIdx.x * PRE_THREADS + threadIdx.x;
    if (i >= n_embd) return;
    const float* xt = x + (size_t) t * HC * n_embd + i;
    float sum = 0.0f;
#pragma unroll
    for (int h = 0; h < HC; ++h) sum = fmaf(xt[(size_t) h * n_embd], s_pre[h], sum);
    cur[(size_t) t * n_embd + i] = sum;
}

// ggml_compute_forward_dsv4_hc_post_f32 (ops.cpp 11557-11630); one thread per (token, i), all hc outputs, so the
// residual values are read before any output is written (out may alias residual).
__global__ void mhc_post_kernel(const float* __restrict__ f, const float* residual, const float* __restrict__ post,
                                const float* __restrict__ comb, float* out, int n_embd) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float* r = residual + (size_t) t * HC * n_embd + i;
    float rv[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) rv[s] = r[(size_t) s * n_embd];
    const float xv = f[(size_t) t * n_embd + i];
    const float* p = post + (size_t) t * HC;
    const float* c = comb + (size_t) t * HC * HC;
    float* o = out + (size_t) t * HC * n_embd + i;
#pragma unroll
    for (int d = 0; d < HC; ++d) {
        float sum = xv * p[d];
#pragma unroll
        for (int s = 0; s < HC; ++s) sum = fmaf(rv[s], c[d + HC * s], sum);
        o[(size_t) d * n_embd] = sum;
    }
}

// build_hc_mean (G:370-378): acc = x0; acc = ggml_add(acc, x_s) for s = 1..hc-1; ggml_scale(acc, 1 / hc)
__global__ void mhc_mean_kernel(const float* __restrict__ x, float* __restrict__ out, int n_embd, float inv_hc) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float* xt = x + (size_t) t * HC * n_embd + i;
    float acc = xt[0];
#pragma unroll
    for (int s = 1; s < HC; ++s) acc = add_rn(acc, xt[(size_t) s * n_embd]);
    out[(size_t) t * n_embd + i] = acc * inv_hc;
}

void check_shape(const MhcShape& s, int n_tok, const char* what) {
    if (s.hc != HC || s.n_embd <= 0 || s.sinkhorn_iters < 1 || n_tok < 1 || n_tok > 65535 || !(s.rms_eps >= 0.0f) ||
        !(s.hc_eps >= 0.0f))
        throw std::invalid_argument(std::string(what) + ": requires hc 4, n_embd > 0, >= 1 Sinkhorn iteration, "
                                    "1..65535 tokens and non-negative eps");
}
void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void mhc_flat_norm(const float* x, float* flat, const MhcShape& shape, int n_tok, void* stream) {
    check_shape(shape, n_tok, "mhc_flat_norm");
    if (!x || !flat) throw std::invalid_argument("mhc_flat_norm: null buffer");
    mhc_flat_norm_kernel<<<(unsigned) n_tok, NORM_THREADS, 0, (cudaStream_t) stream>>>(x, flat, HC * shape.n_embd,
                                                                                        shape.rms_eps);
    check_launch("mhc_flat_norm");
}

void mhc_pre(const float* x, const float* mixes, const float* hc_scale, const float* hc_base, float* cur, float* post,
             float* comb, const MhcShape& shape, int n_tok, void* stream) {
    check_shape(shape, n_tok, "mhc_pre");
    if (!x || !mixes || !hc_scale || !hc_base || !cur || !post || !comb)
        throw std::invalid_argument("mhc_pre: null buffer");
    const dim3 grid((unsigned) ((shape.n_embd + PRE_THREADS - 1) / PRE_THREADS), (unsigned) n_tok);
    mhc_pre_kernel<<<grid, PRE_THREADS, 0, (cudaStream_t) stream>>>(x, mixes, hc_scale, hc_base, cur, post, comb,
                                                                    shape.n_embd, shape.hc_eps, shape.sinkhorn_iters);
    check_launch("mhc_pre");
}

void mhc_post(const float* f, const float* residual, const float* post, const float* comb, float* out,
              const MhcShape& shape, int n_tok, void* stream) {
    check_shape(shape, n_tok, "mhc_post");
    if (!f || !residual || !post || !comb || !out) throw std::invalid_argument("mhc_post: null buffer");
    const dim3 grid((unsigned) ((shape.n_embd + 255) / 256), (unsigned) n_tok);
    mhc_post_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(f, residual, post, comb, out, shape.n_embd);
    check_launch("mhc_post");
}

void mhc_mean(const float* x, float* out, const MhcShape& shape, int n_tok, void* stream) {
    check_shape(shape, n_tok, "mhc_mean");
    if (!x || !out) throw std::invalid_argument("mhc_mean: null buffer");
    const dim3 grid((unsigned) ((shape.n_embd + 255) / 256), (unsigned) n_tok);
    mhc_mean_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, out, shape.n_embd, 1.0f / (float) HC);
    check_launch("mhc_mean");
}

}  // namespace strata::kernels
