// src/kernels/cuda/router_sigmoid.cu - see include/strata/kernels/router_sigmoid.hpp.  Full-precision expf and
// division (no --use_fast_math), so p = sigmoid(logit) follows ggml_vec_sigmoid_f32 to an ulp.
#include "strata/kernels/router_sigmoid.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int LOGIT_TOKENS = 8;      // tokens per pass of the logits kernel (registers per thread)
constexpr int LOGIT_THREADS = 128;   // one block per expert row: 4 warps split the row
constexpr int MAX_K = 32;

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}

__global__ void __launch_bounds__(LOGIT_THREADS) router_logits_kernel(const float* __restrict__ w,
                                                                      const float* __restrict__ x,
                                                                      float* __restrict__ logits, int n_embd,
                                                                      int n_expert, int t0, int nt) {
    __shared__ float part[LOGIT_THREADS / 32][LOGIT_TOKENS];
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float4* w4 = reinterpret_cast<const float4*>(w + (size_t) row * n_embd);
    float acc[LOGIT_TOKENS];
#pragma unroll
    for (int t = 0; t < LOGIT_TOKENS; ++t) acc[t] = 0.0f;
    for (int j = threadIdx.x; j < n_embd / 4; j += LOGIT_THREADS) {
        const float4 wv = w4[j];
#pragma unroll
        for (int t = 0; t < LOGIT_TOKENS; ++t) {
            if (t >= nt) break;
            const float4 xv = reinterpret_cast<const float4*>(x + (size_t) (t0 + t) * n_embd)[j];
            float a = acc[t];
            a = fmaf(wv.x, xv.x, a);
            a = fmaf(wv.y, xv.y, a);
            a = fmaf(wv.z, xv.z, a);
            a = fmaf(wv.w, xv.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < LOGIT_TOKENS; ++t) {
        if (t >= nt) break;
        const float s = warp_sum32(acc[t]);
        if (lane == 0) part[warp][t] = s;
    }
    __syncthreads();
    if ((int) threadIdx.x < nt)
        logits[(size_t) (t0 + threadIdx.x) * n_expert + row] =
            (part[0][threadIdx.x] + part[1][threadIdx.x]) + (part[2][threadIdx.x] + part[3][threadIdx.x]);
}

// One block per token, one thread per expert (blockDim = n_expert rounded up to 32, at most 1024).  Expert e's rank
// is the number of experts whose selection value is larger, or equal with a lower index (ties: lower index first),
// counted against the block's copy in shared memory; the k experts of rank < k write themselves.  No sequential
// arg-max rounds: on RDNA every cross-lane shuffle of such a round costs ~100 cycles, ~1 us per round for 288 experts.
constexpr int TOPK_MAX_EXPERTS = 1024;

__global__ void __launch_bounds__(TOPK_MAX_EXPERTS) router_topk_kernel(const float* __restrict__ logits,
                                                                       const float* __restrict__ bias,
                                                                       int32_t* __restrict__ ids,
                                                                       float* __restrict__ weights, int n_expert, int k,
                                                                       int norm, float scale) {
    __shared__ float s_sel[TOPK_MAX_EXPERTS];
    __shared__ float s_w[MAX_K];
    __shared__ int s_id[MAX_K];
    const int t = blockIdx.x, e = threadIdx.x;
    float p = 0.0f, sel = -INFINITY;
    if (e < n_expert) {
        p = 1.0f / (1.0f + expf(-logits[(size_t) t * n_expert + e]));   // ggml_vec_sigmoid_f32
        sel = bias ? p + bias[e] : p;                                     // probs + exp_probs_b
        s_sel[e] = sel;
    }
    __syncthreads();
    if (e < n_expert) {
        // Every lane walks all n_expert values with the same trip count (no divergence), 16 shared-memory reads in
        // flight per step: one read at a time costs a full LDS latency each, ~12 us for 288 experts on gfx1201.
        int rank = 0;
        int j = 0;
        for (; j + 16 <= n_expert; j += 16) {
            float o[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) o[u] = s_sel[j + u];
#pragma unroll
            for (int u = 0; u < 16; ++u) rank += (o[u] > sel) | ((o[u] == sel) & (j + u < e));
        }
        for (; j < n_expert; ++j) {
            const float o = s_sel[j];
            rank += (o > sel) | ((o == sel) & (j < e));
        }
        if (rank < k) { s_w[rank] = p; s_id[rank] = e; }
    }
    __syncthreads();
    if (e != 0) return;
    float sum = 1.0f;
    if (norm) {
        double acc = 0.0;                       // ggml_vec_sum_f32: ggml_float (double) accumulation, rank order
        for (int r = 0; r < k; ++r) acc += (double) s_w[r];
        sum = fmaxf((float) acc, 6.103515625e-5f);
    }
    for (int r = 0; r < k; ++r) {
        float wv = s_w[r];
        if (norm) wv = wv / sum;
        if (scale != 0.0f && scale != 1.0f) wv = wv * scale;
        ids[(size_t) t * k + r] = s_id[r];
        weights[(size_t) t * k + r] = wv;
    }
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void router_f32_logits(const float* w, const float* x, float* logits, const SigmoidRouterShape& shape, int n_tok,
                       void* stream) {
    if (!w || !x || !logits || shape.n_embd <= 0 || shape.n_embd % 4 != 0 || shape.n_expert <= 0 || n_tok < 1)
        throw std::invalid_argument("router_f32_logits: null buffer, n_embd not a positive multiple of 4, or no rows");
    for (int t0 = 0; t0 < n_tok; t0 += LOGIT_TOKENS) {
        const int nt = n_tok - t0 < LOGIT_TOKENS ? n_tok - t0 : LOGIT_TOKENS;
        router_logits_kernel<<<(unsigned) shape.n_expert, LOGIT_THREADS, 0, (cudaStream_t) stream>>>(
            w, x, logits, shape.n_embd, shape.n_expert, t0, nt);
    }
    check_launch("router_f32_logits");
}

void router_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* weights,
                         const SigmoidRouterShape& shape, int n_tok, void* stream) {
    if (!logits || !ids || !weights || shape.n_expert <= 0 || shape.n_expert > 1024 || shape.k < 1 ||
        shape.k > MAX_K || shape.k > shape.n_expert || n_tok < 1)
        throw std::invalid_argument("router_sigmoid_topk: null buffer, n_expert outside 1..1024 or k outside 1..32");
    const unsigned threads = (unsigned) ((shape.n_expert + 31) / 32 * 32);
    router_topk_kernel<<<(unsigned) n_tok, threads, 0, (cudaStream_t) stream>>>(
        logits, bias, ids, weights, shape.n_expert, shape.k, shape.norm ? 1 : 0, shape.scale);
    check_launch("router_sigmoid_topk");
}

}  // namespace strata::kernels
