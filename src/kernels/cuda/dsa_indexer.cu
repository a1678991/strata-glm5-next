// src/kernels/cuda/dsa_indexer.cu - the DSA layer's k-pool indexer: pooling, per-pool scores, top-k selection.
// See include/strata/kernels/dsa.hpp. Reference: llama.cpp f1cee99 `build_kpool_select` (src/models/glm5-next.cpp
// :766-919) with the fused lightning indexer (ggml-cpu ops.cpp `ggml_compute_forward_lightning_indexer`), and the
// pool layout and masks of `set_input_kpool` (src/llama-memory-hybrid-idx.cpp:791-1036) for one sequence whose
// positions start at 0: pool j is cells kpool*j .. kpool*j + kpool - 1, visible to a token from its last cell on.
//
// Adapted from qsa_select.cu's block top-k (radix select over order-preserving keys, ties to the lower index,
// ascending emission), for 32 indexer heads and per-pool scores; Strata's QSA kernels are not changed.
#include "strata/kernels/dsa.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dsa_indexer: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int MAX_KPOOL = 8;
constexpr int MAX_IDX_DIM = 256;
constexpr int MAX_IDX_HEADS = 64;

// ---- pooling: one block per pool, one thread per channel -------------------------------------------------------
// ggml-cpu, in llama.cpp's order: logits = gate + ape (f32), soft_max over the kpool members (max; expf; the sum in
// double; times (float)(1 / sum)), times the key (f32), summed over the members in double (sum_rows), then the f16
// store of the pooled slot (set_rows). The key | gate rows are the f16 cache rows (get_rows -> f32, exact).
__global__ void pool_kernel(DsaKvCache c, const float* __restrict__ ape, const int32_t* __restrict__ steps, int T,
                            long long j0_host, int n_host, int idx_dim, int kpool) {
    long long j;
    if (steps) {
        const long long pos0 = steps[kStepPos], pos1 = steps[(T - 1) * kStepCount + kStepPos];
        const long long first = pos0 / kpool, end = (pos1 + 1) / kpool;   // pools whose last cell is in [pos0, pos1]
        j = first + blockIdx.x;
        if (j >= end) return;
    } else {
        if ((int) blockIdx.x >= n_host) return;
        j = j0_host + blockIdx.x;
    }
    const int ch = threadIdx.x;
    if (ch >= idx_dim) return;
    const int slot = c.page_table[j];   // page j holds the pool's kpool cells
    const uint16_t* kg = slot >= 0 ? c.kg + (size_t) slot * kpool * 2 * idx_dim : c.host.kg + (size_t) j * kpool * 2 * idx_dim;
    float key[MAX_KPOOL], logit[MAX_KPOOL];
    float mx = -INFINITY;
    for (int k = 0; k < kpool; ++k) {
        key[k] = f32_from_f16(kg[(size_t) k * 2 * idx_dim + ch]);
        logit[k] = f32_from_f16(kg[(size_t) k * 2 * idx_dim + idx_dim + ch]) + ape[(size_t) k * idx_dim + ch];
        mx = fmaxf(mx, logit[k]);
    }
    double sum = 0.0;
    float e[MAX_KPOOL];
    for (int k = 0; k < kpool; ++k) {
        e[k] = expf(logit[k] - mx);
        sum += (double) e[k];
    }
    const float inv = (float) (1.0 / sum);
    double acc = 0.0;
    for (int k = 0; k < kpool; ++k) {
        const float p = e[k] * inv;
        acc += (double) (p * key[k]);
    }
    c.pooled[(size_t) j * idx_dim + ch] = f16_from_f32((float) acc);
}

// ---- scores: a block per (SCORE_POOLS pools, token), a warp per pool at a time, a lane per indexer head ---------
// score = sum over heads, IN HEAD ORDER, of max(q_h . k, 0) * w_h, as ggml-cpu's lightning indexer; the dot is f32.
// The token's queries are staged once per block for SCORE_POOLS = 16 pools; four partial sums per dot.
constexpr int SCORE_WARPS = 8, SCORE_POOLS = 16;

__global__ void __launch_bounds__(SCORE_WARPS * 32) score_kernel(const uint16_t* __restrict__ pooled,
                                                                 const float* __restrict__ iq,
                                                                 const float* __restrict__ w,
                                                                 const int32_t* __restrict__ steps, int n_heads,
                                                                 int idx_dim, long long max_pools,
                                                                 float* __restrict__ scores) {
    extern __shared__ float sm[];
    const int qs = idx_dim + 1;                                   // padded query row: lane h reads row h conflict-free
    float* sq = sm;                                               // [n_heads][idx_dim + 1]
    float* sw = sq + n_heads * qs;                                // [n_heads]
    float* sk = sw + n_heads;                                     // [SCORE_WARPS][idx_dim]
    float* st = sk + SCORE_WARPS * idx_dim;                       // [SCORE_WARPS][n_heads]
    const int tok = blockIdx.y;
    const long long n_bid = steps[tok * kStepCount + kStepNBid];
    const long long j0 = (long long) blockIdx.x * SCORE_POOLS;
    if (j0 >= n_bid) return;                                      // the whole block: uniform
    const float* q = iq + (size_t) tok * n_heads * idx_dim;
    for (int i = threadIdx.x; i < n_heads * idx_dim; i += blockDim.x) sq[(i / idx_dim) * qs + i % idx_dim] = q[i];
    for (int i = threadIdx.x; i < n_heads; i += blockDim.x) sw[i] = w[(size_t) tok * n_heads + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* k = sk + warp * idx_dim;
    float* term = st + warp * n_heads;
    for (long long j = j0 + warp; j < j0 + SCORE_POOLS && j < n_bid; j += SCORE_WARPS) {
        for (int d = lane; d < idx_dim; d += 32) k[d] = f32_from_f16(pooled[(size_t) j * idx_dim + d]);
        __syncwarp();
        for (int h = lane; h < n_heads; h += 32) {
            const float* qh = sq + h * qs;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            int d = 0;
            for (; d + 4 <= idx_dim; d += 4) {
                d0 = fmaf(qh[d], k[d], d0);
                d1 = fmaf(qh[d + 1], k[d + 1], d1);
                d2 = fmaf(qh[d + 2], k[d + 2], d2);
                d3 = fmaf(qh[d + 3], k[d + 3], d3);
            }
            for (; d < idx_dim; ++d) d0 = fmaf(qh[d], k[d], d0);
            const float dot = (d0 + d1) + (d2 + d3);
            term[h] = fmaxf(dot, 0.0f) * sw[h];
        }
        __syncwarp();
        if (lane == 0) {
            float score = 0.0f;
            for (int h = 0; h < n_heads; ++h) score += term[h];
            scores[(size_t) tok * max_pools + j] = score;
        }
        __syncwarp();
    }
}

// ---- selection: one block per token -----------------------------------------------------------------------------
constexpr int SEL_T = 1024;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Block-wide exclusive prefix sum of one int per thread (SEL_T threads); `total` is the block's sum.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

__global__ void __launch_bounds__(SEL_T) select_kernel(const float* __restrict__ scores,
                                                       const int32_t* __restrict__ steps, long long max_pools,
                                                       long long cap, int kpool, int top, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int warp_sums[32];
    __shared__ int s_digit, s_above;
    const int tok = blockIdx.x, t = threadIdx.x;
    const int32_t* stp = steps + tok * kStepCount;
    const long long pos = stp[kStepPos];
    const long long n = stp[kStepNBid];                     // visible (complete) pools
    const int n_tail = (int) ((pos + 1) % kpool);
    int32_t* out = ids + (size_t) tok * cap;
    const long long n_sel_pools = n < top ? n : top;
    if (t < n_tail) out[n_sel_pools * kpool + t] = (int32_t) (pos - n_tail + 1 + t);   // the tail, ascending
    if (n <= top) {                                         // every visible pool: the identity
        for (long long i = t; i < n * kpool; i += SEL_T) out[i] = (int32_t) i;
        return;
    }
    const float* sc = scores + (size_t) tok * max_pools;
    const long long per = (n + SEL_T - 1) / SEL_T;
    const long long b0 = (long long) t * per, b1 = (b0 + per < n) ? b0 + per : n;
    // radix select: the largest key thr with (pools with key >= thr) >= top, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += SEL_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (long long b = b0; b < b1; ++b) {
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (t < 32) {
            // the digit: the highest d with above + (pools in digits > d) + hist[d] >= top, found by warp 0 with a
            // scan over the 256 bins in descending order (lane l holds bins 255 - 8l .. 248 - 8l); the same digit
            // and count as walking the bins down one by one, with d = 0 when no digit reaches `top`
            const int lane = t;
            int loc[8], sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) { loc[k] = hist[255 - 8 * lane - k]; sum += loc[k]; }
            int incl = sum;
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            int cum = above + incl - sum, fd = -1, fc = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (fd < 0) {
                    if (cum + loc[k] >= top) { fd = 255 - 8 * lane - k; fc = cum; }
                    else cum += loc[k];
                }
            }
            const unsigned found = __ballot_sync(0xffffffffu, fd >= 0);
            const int src = found ? __ffs(found) - 1 : 31;
            int d = __shfl_sync(0xffffffffu, fd, src), c = __shfl_sync(0xffffffffu, fc, src);
            if (!found) {   // no digit reaches top: d = 0 with every bin above it counted
                d = 0;
                c = above + __shfl_sync(0xffffffffu, incl, 31) - hist[0];
            }
            if (lane == 0) { s_digit = d; s_above = c; }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int eq_budget = top - above;                      // pools equal to thr that fit, lowest index first
    int gt = 0, eq = 0;
    for (long long b = b0; b < b1; ++b) {
        const uint32_t k = order_key(sc[b]);
        gt += k > thr;
        eq += k == thr;
    }
    int total = 0;
    const int eq_before = block_scan(eq, warp_sums, total);
    int my_eq = eq_budget - eq_before;
    my_eq = my_eq < 0 ? 0 : (my_eq > eq ? eq : my_eq);
    const int wpos0 = block_scan(gt + my_eq, warp_sums, total);
    long long wpos = (long long) wpos0 * kpool;
    int eq_left = my_eq;
    for (long long b = b0; b < b1; ++b) {
        const uint32_t k = order_key(sc[b]);
        bool take = k > thr;
        if (!take && k == thr && eq_left > 0) { take = true; --eq_left; }
        if (take)
            for (int c = 0; c < kpool; ++c) out[wpos++] = (int32_t) (b * kpool + c);
    }
}

void validate(const DsaShapes& s, const char* what) {
    if (s.kpool < 2 || s.kpool > MAX_KPOOL || s.idx_dim <= 0 || s.idx_dim > MAX_IDX_DIM || s.idx_n_head <= 0 ||
        s.idx_n_head > MAX_IDX_HEADS || s.idx_top_k % s.kpool != 0) {
        std::fprintf(stderr, "dsa_indexer: %s: unsupported geometry (kpool %lld, idx_dim %lld, heads %lld)\n", what,
                     (long long) s.kpool, (long long) s.idx_dim, (long long) s.idx_n_head);
        std::exit(1);
    }
}

}  // namespace

void dsa_pool_step(const DsaKvCache& c, const DsaShapes& s, const float* ape, const int32_t* steps, int64_t T,
                   void* stream) {
    validate(s, "pool step");
    if (T <= 0) return;
    // a window of T cells completes at most (T + kpool - 1) / kpool pools
    const unsigned blocks = (unsigned) ((T + s.kpool - 1) / s.kpool);
    pool_kernel<<<blocks, (unsigned) s.idx_dim, 0, (cudaStream_t) stream>>>(c, ape, steps, (int) T, 0, 0,
                                                                            (int) s.idx_dim, (int) s.kpool);
    check("pool step");
}

void dsa_pool_range(const DsaKvCache& c, const DsaShapes& s, const float* ape, int64_t j0, int64_t j1, void* stream) {
    validate(s, "pool range");
    if (j1 <= j0) return;
    const int64_t n = j1 - j0;
    pool_kernel<<<(unsigned) n, (unsigned) s.idx_dim, 0, (cudaStream_t) stream>>>(c, ape, nullptr, 0, (long long) j0,
                                                                                (int) n, (int) s.idx_dim, (int) s.kpool);
    check("pool range");
}

void dsa_index_scores(const DsaKvCache& c, const DsaShapes& s, const float* iq, const float* w, const int32_t* steps,
                      int64_t T, int64_t max_pools, float* scores, void* stream) {
    validate(s, "scores");
    if (T <= 0) return;
    if (max_pools < c.max_pools) {
        std::fprintf(stderr, "dsa_indexer: scores: row stride %lld < the cache's %lld pools\n", (long long) max_pools,
                     (long long) c.max_pools);
        std::exit(1);
    }
    const size_t smem = sizeof(float) * ((size_t) s.idx_n_head * (s.idx_dim + 1) + s.idx_n_head +
                                         (size_t) SCORE_WARPS * s.idx_dim + (size_t) SCORE_WARPS * s.idx_n_head);
    const dim3 grid((unsigned) ((c.max_pools + SCORE_POOLS - 1) / SCORE_POOLS), (unsigned) T);
    score_kernel<<<grid, SCORE_WARPS * 32, smem, (cudaStream_t) stream>>>(c.pooled, iq, w, steps, (int) s.idx_n_head,
                                                                          (int) s.idx_dim, (long long) max_pools,
                                                                          scores);
    check("scores");
}

void dsa_select(const float* scores, const DsaShapes& s, const int32_t* steps, int64_t T, int64_t max_pools,
                int64_t cap, int32_t* ids, void* stream) {
    validate(s, "select");
    if (T <= 0) return;
    if (cap < dsa_sel_cap(s)) {
        std::fprintf(stderr, "dsa_indexer: select: cap %lld < %lld\n", (long long) cap, (long long) dsa_sel_cap(s));
        std::exit(1);
    }
    select_kernel<<<(unsigned) T, SEL_T, 0, (cudaStream_t) stream>>>(scores, steps, (long long) max_pools,
                                                                      (long long) cap, (int) s.kpool,
                                                                      (int) dsa_top_pools(s), ids);
    check("select");
}

}  // namespace strata::kernels
