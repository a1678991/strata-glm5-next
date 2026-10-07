// src/kernels/cuda/kda_decode.cu - GLM-5.3-Flash KDA decode step and verify-window replay.  See
// include/strata/kernels/kda_decode.hpp for the math, the layouts and the reference lines.
//
// The tiling is fused_gdn.cu's: a block owns one head, its 512 threads are (row group of 32 key indices, value
// column), every thread keeps its 32 state values in registers for the whole window, and a row of the head's state
// is one contiguous 512 B load.  KDA differs from the GDN step in three places: the decay is a per-KEY-index vector
// exp(g[i]) (one value per register row instead of one scalar per head), it is applied to the state BEFORE the
// S^T k product (ggml-cpu's order: S[i][:] *= exp(g[i]), then delta = (v - S^T k) * beta), and the gate itself
// (dt bias, A, the lower-bound sigmoid) is computed here from the raw f_b output instead of by a separate kernel.
//
// No --use_fast_math: expf / division / sqrtf are the full-precision ones, so the kernel tracks ggml-cpu's
// arithmetic to F32 rounding (kda_parity measures it).
#include "strata/kernels/kda_decode.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // head_dim: state rows = cols = 128
constexpr int RG = 4;           // row groups per block
constexpr int RPG = S / RG;     // 32 key indices per thread

__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}

// One block per 128-channel head of one part (q: blocks [0, n_head), k: [n_head, 2 n_head), v: the rest), one
// thread per channel, looping over the window's tokens.  ggml_ssm_conv sums the 4 taps in order from 0
// (ops.cpp ggml_compute_forward_ssm_conv_f32), then ggml_silu, then for q and k build_gdn_l2_norm
// (models.h:14-18): rms_norm(x, eps / n) * (1 / sqrt(n)).
template <bool UPDATE>
__global__ void __launch_bounds__(S) kda_conv_l2_kernel(float* __restrict__ hist, const float* __restrict__ qkv,
                                                        const float* __restrict__ wq, const float* __restrict__ wk,
                                                        const float* __restrict__ wv, float* __restrict__ h,
                                                        int n_head, int T, float l2_eps_n, float inv_sqrt_n) {
    __shared__ float part[S / 32];
    const int d_inner = n_head * S, C = 3 * d_inner;
    const int c = blockIdx.x * S + threadIdx.x;
    const int which = (int) blockIdx.x / n_head;            // 0 q, 1 k, 2 v (block-uniform)
    const float* w = (which == 0 ? wq : which == 1 ? wk : wv) + (size_t) (c - which * d_inner) * 4;
    const float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    float h0 = hist[(size_t) c * 3], h1 = hist[(size_t) c * 3 + 1], h2 = hist[(size_t) c * 3 + 2];
    for (int t = 0; t < T; ++t) {
        const float x = qkv[(size_t) t * C + c];
        float sum = 0.0f;
        sum += h0 * w0;
        sum += h1 * w1;
        sum += h2 * w2;
        sum += x * w3;
        h0 = h1;
        h1 = h2;
        h2 = x;
        float y = sum / (1.0f + expf(-sum));
        if (which < 2) {                                     // q and k heads: the L2 norm over the head's 128
            float sq = warp_sum32(y * y);
            __syncthreads();                                 // part[] of the previous token is consumed
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = (part[0] + part[1]) + (part[2] + part[3]);
            const float scale = 1.0f / sqrtf(ss / (float) S + l2_eps_n);
            y = (y * scale) * inv_sqrt_n;
        }
        h[(size_t) t * C + c] = y;
    }
    if (UPDATE) {
        hist[(size_t) c * 3] = h0;
        hist[(size_t) c * 3 + 1] = h1;
        hist[(size_t) c * 3 + 2] = h2;
    }
}

__global__ void kda_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0 .. x_{n-1}]
        seq[j] = src < 3 ? hist[(size_t) c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[(size_t) c * 3] = seq[0];
    hist[(size_t) c * 3 + 1] = seq[1];
    hist[(size_t) c * 3 + 2] = seq[2];
}

// MODE 0: decode (y for every token, state written); 1: verify (y only); 2: commit (*n_keep tokens, state only).
template <int MODE>
__global__ void __launch_bounds__(S * RG) kda_step_kernel(float* __restrict__ state, const float* __restrict__ hbuf,
                                                          const float* __restrict__ fb, const float* __restrict__ dtb,
                                                          const float* __restrict__ ssm_a,
                                                          const float* __restrict__ b_raw,
                                                          const float* __restrict__ gb,
                                                          const float* __restrict__ onw, float* __restrict__ y,
                                                          int n_head, int T, const int32_t* __restrict__ n_keep,
                                                          float lower_bound, float rms_eps, float scale) {
    __shared__ float sk[S], sq[S], seg[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;                 // value index j
    const int rg = threadIdx.y;                  // key indices [rg * 32, rg * 32 + 32)
    const int d_inner = n_head * S, C = 3 * d_inner;
    const int n = MODE == 2 ? min(*n_keep, T) : T;
    float s[RPG];
    float* base = state + ((size_t) head * S + (size_t) rg * RPG) * S + col;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[(size_t) r * S];
    const float A = ssm_a[head];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();                         // the previous token is done with sk/sq/seg/red/wsum
        if (rg == 0) {
            sk[col] = ht[d_inner + head * S + col];
        } else if (rg == 1) {
            if (MODE != 2) sq[col] = ht[head * S + col];
        } else if (rg == 2) {
            // G:707-721: g1 = (f_b(f_a x) + dt_b) * A; g = lb * sigmoid(-g1).  ggml_scale(-1) is an exact
            // negation, so sigmoid(-g1) = 1 / (1 + exp(g1)).  The recurrence uses exp(g) (ops.cpp KDA branch).
            const int ch = head * S + col;
            const float g1 = (fb[(size_t) t * d_inner + ch] + dtb[ch]) * A;
            const float sg = 1.0f / (1.0f + expf(g1));
            seg[col] = expf(sg * lower_bound);
        }
        __syncthreads();
        const float beta = 1.0f / (1.0f + expf(-b_raw[(size_t) t * n_head + head]));
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] *= seg[rg * RPG + r];           // S[i][:] *= exp(g[i])
            kv = fmaf(s[r], sk[rg * RPG + r], kv);
        }
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = (red[0][col] + red[1][col]) + (red[2][col] + red[3][col]);
        const float delta = (ht[2 * d_inner + head * S + col] - kv_col) * beta;
        if (MODE == 2) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(sk[rg * RPG + r], delta, s[r]);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(sk[rg * RPG + r], delta, s[r]);   // S[i][j] += k[i] * delta[j]
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();                         // every thread has read red[] for kv_col
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f;
        if (rg == 0) {
            oc = ((red[0][col] + red[1][col]) + (red[2][col] + red[3][col])) * scale;
            const float part = warp_sum32(oc * oc);
            if ((col & 31) == 0) wsum[col >> 5] = part;
        }
        __syncthreads();
        if (rg == 0) {
            // build_norm(o, ssm_o_norm, RMS) per head, then * sigmoid(g_b(g_a x)) (G:750-756)
            const float ss = (wsum[0] + wsum[1]) + (wsum[2] + wsum[3]);
            const float sc = 1.0f / sqrtf(ss / (float) S + rms_eps);
            const int ch = head * S + col;
            const float z = gb[(size_t) t * d_inner + ch];
            const float gate = 1.0f / (1.0f + expf(-z));
            y[(size_t) t * d_inner + ch] = ((oc * sc) * onw[col]) * gate;
        }
    }
    if (MODE != 1) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[(size_t) r * S] = s[r];
    }
}

void check_shape(const KdaShape& shape, const char* what) {
    if (shape.head_dim != S || shape.n_head <= 0 || shape.n_head > 65535 || !std::isfinite(shape.gate_lower_bound) ||
        !(shape.rms_eps >= 0.0f) || !(shape.l2_eps >= 0.0f))
        throw std::invalid_argument(std::string(what) + ": requires head_dim 128, 1..65535 heads, a finite gate "
                                    "lower bound and non-negative eps");
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void kda_conv_l2(float* history, const float* qkv, const float* conv_wq, const float* conv_wk, const float* conv_wv,
                 float* h, const KdaShape& shape, int n_tok, bool update_history, void* stream) {
    check_shape(shape, "kda_conv_l2");
    if (!history || !qkv || !conv_wq || !conv_wk || !conv_wv || !h || n_tok < 1 || n_tok > kVerifyMaxT)
        throw std::invalid_argument("kda_conv_l2: null buffer or n_tok outside 1..8");
    const float n = (float) S;
    // build_gdn_l2_norm: ggml_rms_norm(x, eps / n) then ggml_scale(1 / sqrtf(n)), both F32 constants
    const float l2_eps_n = shape.l2_eps / n, inv_sqrt_n = 1.0f / sqrtf(n);
    const dim3 grid((unsigned) (3 * shape.n_head));
    if (update_history)
        kda_conv_l2_kernel<true><<<grid, S, 0, (cudaStream_t) stream>>>(history, qkv, conv_wq, conv_wk, conv_wv, h,
                                                                         shape.n_head, n_tok, l2_eps_n, inv_sqrt_n);
    else
        kda_conv_l2_kernel<false><<<grid, S, 0, (cudaStream_t) stream>>>(history, qkv, conv_wq, conv_wk, conv_wv, h,
                                                                          shape.n_head, n_tok, l2_eps_n, inv_sqrt_n);
    check_launch("kda_conv_l2");
}

void kda_conv_commit(float* history, const float* qkv, const KdaShape& shape, const int32_t* n_keep, void* stream) {
    check_shape(shape, "kda_conv_commit");
    if (!history || !qkv || !n_keep) throw std::invalid_argument("kda_conv_commit: null buffer");
    const int C = 3 * shape.d_inner();
    kda_conv_commit_kernel<<<(unsigned) ((C + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv, C, n_keep);
    check_launch("kda_conv_commit");
}

void kda_step_norm(float* state, const float* h, const float* fb, const float* dt_bias, const float* ssm_a,
                   const float* b_raw, const float* gb, const float* o_norm_w, float* y, const KdaShape& shape,
                   int n_tok, bool write_state, void* stream) {
    check_shape(shape, "kda_step_norm");
    if (!state || !h || !fb || !dt_bias || !ssm_a || !b_raw || !gb || !o_norm_w || !y || n_tok < 1 ||
        n_tok > kVerifyMaxT)
        throw std::invalid_argument("kda_step_norm: null buffer or n_tok outside 1..8");
    const float scale = 1.0f / sqrtf((float) S);   // ops.cpp: scale = 1.0f / sqrtf((float) S_v)
    const dim3 grid((unsigned) shape.n_head), block(S, RG);
    if (write_state)
        kda_step_kernel<0><<<grid, block, 0, (cudaStream_t) stream>>>(state, h, fb, dt_bias, ssm_a, b_raw, gb,
                                                                      o_norm_w, y, shape.n_head, n_tok, nullptr,
                                                                      shape.gate_lower_bound, shape.rms_eps, scale);
    else
        kda_step_kernel<1><<<grid, block, 0, (cudaStream_t) stream>>>(state, h, fb, dt_bias, ssm_a, b_raw, gb,
                                                                      o_norm_w, y, shape.n_head, n_tok, nullptr,
                                                                      shape.gate_lower_bound, shape.rms_eps, scale);
    check_launch("kda_step_norm");
}

void kda_step_commit(float* state, const float* h, const float* fb, const float* dt_bias, const float* ssm_a,
                     const float* b_raw, const KdaShape& shape, const int32_t* n_keep, void* stream) {
    check_shape(shape, "kda_step_commit");
    if (!state || !h || !fb || !dt_bias || !ssm_a || !b_raw || !n_keep)
        throw std::invalid_argument("kda_step_commit: null buffer");
    const float scale = 1.0f / sqrtf((float) S);
    kda_step_kernel<2><<<dim3((unsigned) shape.n_head), dim3(S, RG), 0, (cudaStream_t) stream>>>(
        state, h, fb, dt_bias, ssm_a, b_raw, nullptr, nullptr, nullptr, shape.n_head, kVerifyMaxT, n_keep,
        shape.gate_lower_bound, shape.rms_eps, scale);
    check_launch("kda_step_commit");
}

}  // namespace strata::kernels
