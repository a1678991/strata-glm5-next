// src/kernels/cuda/mla_decode.cu - the DSA layer's MLA decode step: norms, the absorbed per-head products, the
// sparse attention over the selected latent cells, and the composed step. See include/strata/kernels/dsa.hpp.
// Reference: llama.cpp f1cee99 `build_dsa_layer` (src/models/glm5-next.cpp:923-1013), gather path (G:965-990),
// with ggml-cpu's arithmetic for the norms (ops.cpp `ggml_compute_forward_rms_norm_f32`, `_norm_f32`).
#include "strata/kernels/dsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mla_decode: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
[[noreturn]] void refuse(const char* what) {
    std::fprintf(stderr, "mla_decode: %s\n", what);
    std::exit(1);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---- norms: one block per row ----------------------------------------------------------------------------------
constexpr int NORM_T = 256;

__device__ double block_sum_d(double v, double* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = NORM_T / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const double r = red[0];
    __syncthreads();
    return r;
}

// ggml_rms_norm then ggml_mul: sum of the f32 squares in double, mean = (float)(sum / n), scale = 1 / sqrtf(mean +
// eps), y = (x * scale) * w.
__global__ void __launch_bounds__(NORM_T) rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                          float eps, int n, float* __restrict__ y) {
    __shared__ double red[NORM_T];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += NORM_T) {
        const float sq = xr[i] * xr[i];
        s += (double) sq;
    }
    s = block_sum_d(s, red);
    const float mean = (float) (s / (double) n);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int i = threadIdx.x; i < n; i += NORM_T) {
        const float v = xr[i] * scale;
        yr[i] = v * w[i];
    }
}

// ggml_norm then ggml_mul, ggml_add: mean = (float)(double sum) / n, the centred squares summed in double / n,
// scale = 1 / sqrtf(var + eps), y = ((x - mean) * scale) * w + b.
__global__ void __launch_bounds__(NORM_T) layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                            const float* __restrict__ b, float eps, int n,
                                                            float* __restrict__ y) {
    __shared__ double red[NORM_T];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    double s = 0.0;
    for (int i = threadIdx.x; i < n; i += NORM_T) s += (double) xr[i];
    s = block_sum_d(s, red);
    const float mean = (float) s / (float) n;
    double v2 = 0.0;
    for (int i = threadIdx.x; i < n; i += NORM_T) {
        const float d = xr[i] - mean;
        const float sq = d * d;
        v2 += (double) sq;
    }
    v2 = block_sum_d(v2, red);
    const float var = (float) (v2 / (double) n);
    const float scale = 1.0f / sqrtf(var + eps);
    for (int i = threadIdx.x; i < n; i += NORM_T) {
        const float d = (xr[i] - mean) * scale;
        const float m = d * w[i];
        yr[i] = m + b[i];
    }
}

// ---- an F32 weight: one 256-thread block per (output, column), the dot split over its 8 warps ----------------
__global__ void __launch_bounds__(256) gemv_f32_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out, float scale) {
    __shared__ float red[8];
    const int o = blockIdx.x, col = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* wr = w + (size_t) o * n_in;
    const float* xc = x + (size_t) col * n_in;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += 256) acc = fmaf(wr[i], xc[i], acc);
    acc = warp_sum(acc);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < 8; ++k) s += red[k];
        y[(size_t) col * n_out + o] = s * scale;
    }
}

// ---- per-head Q8_0 x Q8_1 -----------------------------------------------------------------------------------
// One lane per 32-value block of a row, nb = n_in / 32 lanes per row (8 for k_b, 16 for v_b), 256 / nb rows per
// 256-thread block. The block's rows are contiguous GGUF bytes (rows * 34 * nb, a multiple of 4), staged in shared
// memory with coalesced 4-byte loads; a lane then takes its block's scale and 32 codes from there, its exact
// integer dot with the activation's Q8_1 block, and the block's two scales as ggml's vec_dot_q8_0_q8_0 does
// (sumi * (d_w * d_x)); the nb lanes of a row sum their blocks in a shuffle tree.
struct Q81 {
    uint16_t d, s;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "q8_1 block");

__device__ __forceinline__ int dot4(int a, int b) {
    int s = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) s += (int) (int8_t) (a >> (8 * k)) * (int) (int8_t) (b >> (8 * k));
    return s;
}

constexpr int GQ_THREADS = 256, GQ_BYTES = GQ_THREADS * 34;   // one 34-byte Q8_0 block per thread

__global__ void __launch_bounds__(GQ_THREADS) grouped_q8_0_kernel(const uint8_t* __restrict__ w,
                                                                  const Q81* __restrict__ x, float* __restrict__ y,
                                                                  int nb, int rows, int n_groups, int ncols) {
    __shared__ __align__(16) uint32_t sw[GQ_BYTES / 4];
    const int rows_per_block = GQ_THREADS / nb;
    const long long r0 = (long long) blockIdx.x * rows_per_block;   // first global row (group * rows + row)
    const long long total_rows = (long long) rows * n_groups;
    const int n_rows = (int) min((long long) rows_per_block, total_rows - r0);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(w + r0 * nb * 34);
    for (int i = threadIdx.x; i < n_rows * nb * 34 / 4; i += GQ_THREADS) sw[i] = src[i];
    __syncthreads();
    const int lr = threadIdx.x / nb, b = threadIdx.x % nb;
    if (lr >= n_rows) return;   // whole rows only: the shuffle groups below stay complete
    const long long gr = r0 + lr;
    const int g = (int) (gr / rows), r = (int) (gr % rows);
    const uint16_t* blk = reinterpret_cast<const uint16_t*>(sw) + (lr * nb + b) * 17;   // 34 bytes = 17 halves
    const float dw = f32_from_f16(blk[0]);
    int wq[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) wq[k] = (int) ((uint32_t) blk[1 + 2 * k] | ((uint32_t) blk[2 + 2 * k] << 16));
    for (int c = 0; c < ncols; ++c) {
        const Q81& xb = x[((size_t) c * n_groups + g) * nb + b];
        const int* xq = reinterpret_cast<const int*>(xb.qs);
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) sumi += dot4(wq[k], xq[k]);
        float v = (float) sumi * (dw * f32_from_f16(xb.d));
        for (int o = nb / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (b == 0) y[((size_t) c * n_groups + g) * rows + r] = v;
    }
}

// ---- attention over the selected cells ------------------------------------------------------------------------
// grid (chunks of A_CHUNK selected cells, head groups of A_HG, tokens). A block reads its chunk's latent rows once
// for A_HG heads: the scores, the chunk's max and exp-sum per head, and p . V with thread t owning dimensions t and
// t + 256. int8 (the production format): the chunk is staged in shared memory and each thread computes one whole
// (head, cell) dot product - no cross-lane reductions, which were two thirds of the kernel's time on gfx1201. f16
// (a parity mode): one warp per cell, lane l holding dimensions l, l + 32, ..., reduced across the warp. A merge
// kernel combines the chunks with the log-sum-exp rescale (qsa_decode_attn.cu's scheme).
// A_CHUNK = 32: 16 KiB of staged codes beside 16 KiB of queries (CUDA's static limit is 48 KiB), and 520 blocks for
// one token's 2,051 cells.
constexpr int A_CHUNK = 32, A_HG = 8, A_THREADS = 256, A_WARPS = A_THREADS / 32;
static_assert(A_CHUNK % 32 == 0, "the softmax phase takes A_CHUNK / 32 cells per lane");

template <int FMT, int PER>
__device__ __forceinline__ float load_one(const DsaKvCache& c, long long row, int d) {
    constexpr int D = PER * 32;
    if constexpr (FMT == kDsaLatInt8)
        return (float) c.lat_q[row * D + d] * f32_from_f16(c.lat_scale[row * (D / DSA_LAT_GROUP) + d / DSA_LAT_GROUP]);
    else
        return f32_from_f16(c.lat_f16[row * D + d]);
}

template <int FMT, int PER>
__global__ void __launch_bounds__(A_THREADS) attend_chunk_kernel(DsaKvCache c, const float* __restrict__ q_abs,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, long long cap,
                                                                 int n_head, int kpool, float scale,
                                                                 float* __restrict__ part_acc,
                                                                 float* __restrict__ part_m,
                                                                 float* __restrict__ part_l, int n_chunks) {
    constexpr int D = PER * 32;
    constexpr int DPT = (D + A_THREADS - 1) / A_THREADS;   // value dims per thread
    constexpr bool STAGE = FMT == kDsaLatInt8;
    constexpr int G = D / DSA_LAT_GROUP;
    __shared__ __align__(16) float sq[A_HG][D];
    __shared__ float sp[A_HG][A_CHUNK];
    __shared__ __align__(16) float spt[A_CHUNK][A_HG];   // the probabilities, cell-major: two float4 per cell
    __shared__ long long srow[A_CHUNK];
    // staged rows are D + 4 bytes apart (an odd number of 32-bit words: the 32 cells of a wave read 32 banks) and
    // the scale rows G + 1 floats apart, for the same reason
    constexpr int CW = STAGE ? D / 4 + 1 : 1;   // words per staged row
    __shared__ uint32_t scode[STAGE ? A_CHUNK : 1][CW];
    __shared__ float sscale[STAGE ? A_CHUNK : 1][STAGE ? G + 1 : 1];
    const int chunk = blockIdx.x, hg = blockIdx.y, tok = blockIdx.z;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int width = steps[tok * kStepCount + kStepWidth];
    const int c0 = chunk * A_CHUNK;
    const int n_here = min(A_CHUNK, width - c0);
    const size_t pbase = ((size_t) tok * n_chunks + chunk) * n_head + (size_t) hg * A_HG;   // this block's 8 heads
    if (n_here <= 0) {
        if (t < A_HG) { part_m[pbase + t] = -FLT_MAX; part_l[pbase + t] = 0.0f; }
        return;
    }
    const float* qsrc = q_abs + ((size_t) tok * n_head + (size_t) hg * A_HG) * D;
    for (int i = t; i < A_HG * D / 4; i += A_THREADS)
        reinterpret_cast<float4*>(&sq[0][0])[i] = reinterpret_cast<const float4*>(qsrc)[i];
    if (t < A_CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[(size_t) tok * cap + c0 + t];
            const int slot = c.page_table[cell / kpool];
            if (slot >= 0) r = (long long) slot * kpool + cell % kpool;   // a non-resident page is masked
        }
        srow[t] = r;
    }
    __syncthreads();
    if constexpr (STAGE) {
        static_assert(A_CHUNK == 32 && A_HG == A_WARPS, "one wave per head, one lane per cell");
        for (int i = t; i < A_CHUNK * (D / 16); i += A_THREADS) {
            const int cc = i / (D / 16), v = i % (D / 16);
            if (cc < n_here && srow[cc] >= 0) {
                const uint4 w = *reinterpret_cast<const uint4*>(c.lat_q + srow[cc] * D + v * 16);
                scode[cc][v * 4 + 0] = w.x;
                scode[cc][v * 4 + 1] = w.y;
                scode[cc][v * 4 + 2] = w.z;
                scode[cc][v * 4 + 3] = w.w;
            }
        }
        for (int i = t; i < A_CHUNK * G; i += A_THREADS) {
            const int cc = i / G, g = i % G;
            if (cc < n_here && srow[cc] >= 0) sscale[cc][g] = f32_from_f16(c.lat_scale[srow[cc] * G + g]);
        }
        __syncthreads();
        // scores: thread (wave h, lane cc) takes the whole dot of head h and cell cc, no cross-lane reduction
        const int h = warp, cc = lane;
        if (cc >= n_here || srow[cc] < 0) {
            sp[h][cc] = -FLT_MAX;
        } else {
            // four accumulators per 64-value group, then the groups: a short tree, like ggml's SIMD dot
            float sacc = 0.0f;
            for (int g = 0; g < G; ++g) {
                const float sc = sscale[cc][g];
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                for (int w = 0; w < DSA_LAT_GROUP / 4; ++w) {
                    const uint32_t word = scode[cc][g * (DSA_LAT_GROUP / 4) + w];
                    const float4 qv = *reinterpret_cast<const float4*>(&sq[h][g * DSA_LAT_GROUP + 4 * w]);
                    a0 = fmaf((float) (int8_t) (word & 0xffu) * sc, qv.x, a0);
                    a1 = fmaf((float) (int8_t) ((word >> 8) & 0xffu) * sc, qv.y, a1);
                    a2 = fmaf((float) (int8_t) ((word >> 16) & 0xffu) * sc, qv.z, a2);
                    a3 = fmaf((float) (int8_t) (word >> 24) * sc, qv.w, a3);
                }
                sacc += (a0 + a1) + (a2 + a3);
            }
            sp[h][cc] = sacc * scale;
        }
    }
    for (int cc = warp; cc < (STAGE ? 0 : A_CHUNK); cc += A_WARPS) {
        if (cc >= n_here || srow[cc] < 0) {
            if (lane < A_HG) sp[lane][cc] = -FLT_MAX;
            continue;
        }
        float k[PER];   // dimension lane + 32 i
#pragma unroll
        for (int i = 0; i < PER; ++i) k[i] = load_one<FMT, PER>(c, srow[cc], lane + 32 * i);
#pragma unroll
        for (int h = 0; h < A_HG; ++h) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < PER; ++i) s = fmaf(k[i], sq[h][lane + 32 * i], s);
            s = warp_sum(s);
            if (lane == 0) sp[h][cc] = s * scale;
        }
    }
    __syncthreads();
    for (int h = warp; h < A_HG; h += A_WARPS) {
        constexpr int PL = A_CHUNK / 32;
        float v[PL];
        float m = -FLT_MAX;
#pragma unroll
        for (int j = 0; j < PL; ++j) {
            v[j] = sp[h][lane + 32 * j];
            m = fmaxf(m, v[j]);
        }
        m = warp_max(m);
        float l = 0.0f;
#pragma unroll
        for (int j = 0; j < PL; ++j) {
            const int cc = lane + 32 * j;
            const float e = (cc < n_here && srow[cc] >= 0) ? expf(v[j] - m) : 0.0f;
            spt[cc][h] = e;
            l += e;
        }
        l = warp_sum(l);
        if (lane == 0) { part_m[pbase + h] = m; part_l[pbase + h] = l; }
    }
    __syncthreads();
    float acc[A_HG][DPT];
#pragma unroll
    for (int h = 0; h < A_HG; ++h)
#pragma unroll
        for (int j = 0; j < DPT; ++j) acc[h][j] = 0.0f;
    static_assert(A_HG == 8, "two float4 of probabilities per cell");
    for (int cc = 0; cc < n_here; ++cc) {
        const long long row = srow[cc];
        if (row < 0) continue;
        const float4 p0 = *reinterpret_cast<const float4*>(&spt[cc][0]);
        const float4 p1 = *reinterpret_cast<const float4*>(&spt[cc][4]);
        const float p[A_HG] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const int d = t + j * A_THREADS;
            if (d >= D) break;
            float v;
            if constexpr (STAGE)
                v = (float) (int8_t) ((scode[cc][d >> 2] >> (8 * (d & 3))) & 0xffu) * sscale[cc][d / DSA_LAT_GROUP];
            else v = load_one<FMT, PER>(c, row, d);
#pragma unroll
            for (int h = 0; h < A_HG; ++h) acc[h][j] = fmaf(p[h], v, acc[h][j]);
        }
    }
#pragma unroll
    for (int h = 0; h < A_HG; ++h)
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const int d = t + j * A_THREADS;
            if (d < D) part_acc[(pbase + h) * D + d] = acc[h][j];
        }
}

// grid (n_head, tokens, D / M_DIMS): the chunks' weights exp(m_c - M) once per block in shared memory, then each
// thread folds its dimension over the chunks, four chunks' loads in flight at a time.
constexpr int M_THREADS = 128, M_MAX_CHUNKS = 1024;
__global__ void __launch_bounds__(M_THREADS) attend_merge_kernel(const float* __restrict__ part_acc,
                                                                 const float* __restrict__ part_m,
                                                                 const float* __restrict__ part_l, int n_chunks,
                                                                 int n_head, int D, float* __restrict__ o) {
    __shared__ float sw[M_MAX_CHUNKS];
    __shared__ float s_M, s_L;
    const int h = blockIdx.x, tok = blockIdx.y;
    const size_t base = (size_t) tok * n_chunks * n_head + h;   // + ch * n_head
    if (threadIdx.x < 32) {
        float M = -FLT_MAX;
        for (int ch = threadIdx.x; ch < n_chunks; ch += 32) M = fmaxf(M, part_m[base + (size_t) ch * n_head]);
        M = warp_max(M);
        if (threadIdx.x == 0) s_M = M;
    }
    __syncthreads();
    for (int ch = threadIdx.x; ch < n_chunks; ch += M_THREADS) {
        const float m = part_m[base + (size_t) ch * n_head];
        sw[ch] = m == -FLT_MAX ? 0.0f : expf(m - s_M);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float L = 0.0f;
        for (int ch = 0; ch < n_chunks; ++ch) L = fmaf(part_l[base + (size_t) ch * n_head], sw[ch], L);
        s_L = L;
    }
    __syncthreads();
    const int d = blockIdx.z * M_THREADS + threadIdx.x;
    if (d >= D) return;
    float acc = 0.0f;
    int ch = 0;
    for (; ch + 4 <= n_chunks; ch += 4) {
        const float a0 = part_acc[(base + (size_t) ch * n_head) * D + d];
        const float a1 = part_acc[(base + (size_t) (ch + 1) * n_head) * D + d];
        const float a2 = part_acc[(base + (size_t) (ch + 2) * n_head) * D + d];
        const float a3 = part_acc[(base + (size_t) (ch + 3) * n_head) * D + d];
        acc = fmaf(a0, sw[ch], acc);
        acc = fmaf(a1, sw[ch + 1], acc);
        acc = fmaf(a2, sw[ch + 2], acc);
        acc = fmaf(a3, sw[ch + 3], acc);
    }
    for (; ch < n_chunks; ++ch) acc = fmaf(part_acc[(base + (size_t) ch * n_head) * D + d], sw[ch], acc);
    o[((size_t) tok * n_head + h) * D + d] = s_L > 0.0f ? acc / s_L : 0.0f;
}

int64_t attend_chunks(int64_t cap) { return (cap + A_CHUNK - 1) / A_CHUNK; }

uint64_t align256(uint64_t n) { return (n + 255) & ~255ull; }
struct Cursor {
    uint8_t* p;
    uint64_t used = 0;
    template <typename T>
    T* take(uint64_t count) {
        T* r = p ? (T*) (p + used) : nullptr;
        used = align256(used + count * sizeof(T));
        return r;
    }
};

}  // namespace

void dsa_rms_norm(const float* x, const float* w, float eps, int64_t n, int64_t rows, float* y, void* stream) {
    if (rows <= 0) return;
    rms_norm_kernel<<<(unsigned) rows, NORM_T, 0, (cudaStream_t) stream>>>(x, w, eps, (int) n, y);
    check("rms_norm");
}

void dsa_layer_norm(const float* x, const float* w, const float* b, float eps, int64_t n, int64_t rows, float* y,
                    void* stream) {
    if (rows <= 0) return;
    layer_norm_kernel<<<(unsigned) rows, NORM_T, 0, (cudaStream_t) stream>>>(x, w, b, eps, (int) n, y);
    check("layer_norm");
}

void dsa_gemv_f32(const float* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t ncols, float scale,
                  void* stream) {
    if (n_out <= 0 || ncols <= 0) return;
    gemv_f32_kernel<<<dim3((unsigned) n_out, (unsigned) ncols), 256, 0, (cudaStream_t) stream>>>(w, x, y, (int) n_in,
                                                                                               (int) n_out, scale);
    check("gemv_f32");
}

void dsa_grouped_q8_0(const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t rows, int64_t n_groups,
                      int64_t ncols, void* stream) {
    const int64_t nb = n_in / 32;
    if (n_in % 32 != 0 || (nb != 8 && nb != 16 && nb != 32)) refuse("grouped Q8_0: n_in must be 256, 512 or 1024");
    if (ncols < 1 || ncols > DSA_MAX_T) refuse("grouped Q8_0: 1 <= ncols <= 8");
    const int64_t rows_per_block = GQ_THREADS / nb, total = rows * n_groups;
    grouped_q8_0_kernel<<<(unsigned) ((total + rows_per_block - 1) / rows_per_block), GQ_THREADS, 0,
                          (cudaStream_t) stream>>>((const uint8_t*) w, (const Q81*) x_q8_1, y, (int) nb, (int) rows,
                                                   (int) n_groups, (int) ncols);
    check("grouped_q8_0");
}

uint64_t dsa_attend_scratch_floats(int64_t cap, int64_t T, const DsaShapes& s) {
    return (uint64_t) T * attend_chunks(cap) * s.n_head * (s.kv_lora + 2) + 64;
}

void dsa_attend(const DsaKvCache& c, const DsaShapes& s, const float* q_abs, const int32_t* ids, const int32_t* steps,
                int64_t T, int64_t cap, float scale, float* scratch, float* o, void* stream) {
    if (T <= 0) return;
    if (s.n_head % A_HG != 0) refuse("attend: n_head must be a multiple of 8");
    if (s.kv_lora != 512 && s.kv_lora != 256) refuse("attend: kv_lora must be 256 or 512");
    if (T > 65535) refuse("attend: too many tokens");
    const int n_chunks = (int) attend_chunks(cap);
    const int64_t D = s.kv_lora;
    float* part_acc = scratch;
    float* part_m = part_acc + (size_t) T * n_chunks * s.n_head * D;
    float* part_l = part_m + (size_t) T * n_chunks * s.n_head;
    const dim3 grid((unsigned) n_chunks, (unsigned) (s.n_head / A_HG), (unsigned) T);
    cudaStream_t st = (cudaStream_t) stream;
    const int H = (int) s.n_head, K = (int) s.kpool;
    if (c.fmt == kDsaLatInt8) {
        if (D == 512)
            attend_chunk_kernel<kDsaLatInt8, 16><<<grid, A_THREADS, 0, st>>>(c, q_abs, ids, steps, cap, H, K, scale,
                                                                             part_acc, part_m, part_l, n_chunks);
        else
            attend_chunk_kernel<kDsaLatInt8, 8><<<grid, A_THREADS, 0, st>>>(c, q_abs, ids, steps, cap, H, K, scale,
                                                                            part_acc, part_m, part_l, n_chunks);
    } else {
        if (D == 512)
            attend_chunk_kernel<kDsaLatF16, 16><<<grid, A_THREADS, 0, st>>>(c, q_abs, ids, steps, cap, H, K, scale,
                                                                            part_acc, part_m, part_l, n_chunks);
        else
            attend_chunk_kernel<kDsaLatF16, 8><<<grid, A_THREADS, 0, st>>>(c, q_abs, ids, steps, cap, H, K, scale,
                                                                           part_acc, part_m, part_l, n_chunks);
    }
    check("attend chunks");
    if (n_chunks > M_MAX_CHUNKS) refuse("attend: too many chunks for the merge");
    attend_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) T, (unsigned) ((D + M_THREADS - 1) / M_THREADS)),
                          M_THREADS, 0, st>>>(part_acc, part_m, part_l, n_chunks, H, (int) D, o);
    check("attend merge");
}

// ================================ the composed step ================================

namespace {
uint64_t q81_bytes(uint64_t n) { return n / 32 * 36; }
uint64_t xq_elems(const DsaShapes& s, int64_t T) {
    uint64_t m = (uint64_t) s.n_embd;
    if ((uint64_t) s.q_lora > m) m = s.q_lora;
    if ((uint64_t) (s.n_head * s.head_k) > m) m = s.n_head * s.head_k;
    if ((uint64_t) (s.n_head * s.kv_lora) > m) m = s.n_head * s.kv_lora;
    if ((uint64_t) (s.n_head * s.head_v) > m) m = s.n_head * s.head_v;
    return m * (uint64_t) T;
}
uint64_t carve_scratch(void* base, const DsaShapes& s, int64_t T, int64_t max_pools, DsaScratch& sc) {
    Cursor c{(uint8_t*) base};
    const int64_t cap = dsa_sel_cap(s);
    sc.xq = c.take<uint8_t>(q81_bytes(xq_elems(s, T)));
    sc.qa = c.take<float>((uint64_t) T * s.q_lora);
    sc.qr = c.take<float>((uint64_t) T * s.q_lora);
    sc.q = c.take<float>((uint64_t) T * s.n_head * s.head_k);
    sc.q_abs = c.take<float>((uint64_t) T * s.n_head * s.kv_lora);
    sc.kv = c.take<float>((uint64_t) T * s.kv_lora);
    sc.latent = c.take<float>((uint64_t) T * s.kv_lora);
    sc.ik_raw = c.take<float>((uint64_t) T * s.idx_dim);
    sc.ik = c.take<float>((uint64_t) T * s.idx_dim);
    sc.ig = c.take<float>((uint64_t) T * s.idx_dim);
    sc.iw = c.take<float>((uint64_t) T * s.idx_n_head);
    sc.iq = c.take<float>((uint64_t) T * s.idx_n_head * s.idx_dim);
    sc.scores = c.take<float>((uint64_t) T * max_pools);
    sc.ids = c.take<int32_t>((uint64_t) T * cap);
    sc.attn = c.take<float>(dsa_attend_scratch_floats(cap, T, s));
    sc.o_lat = c.take<float>((uint64_t) T * s.n_head * s.kv_lora);
    sc.o = c.take<float>((uint64_t) T * s.n_head * s.head_v);
    sc.cap = cap;
    sc.max_pools = max_pools;
    sc.max_t = T;
    return c.used;
}
}  // namespace

uint64_t dsa_scratch_bytes(const DsaShapes& s, int64_t max_t, int64_t max_pools) {
    DsaScratch sc;
    return carve_scratch(nullptr, s, max_t, max_pools, sc) + 256;
}

void dsa_scratch_carve(void* base, const DsaShapes& s, int64_t max_t, int64_t max_pools, DsaScratch& sc) {
    carve_scratch(base, s, max_t, max_pools, sc);
}

void dsa_decode(const DsaWeights& wt, const DsaShapes& s, const DsaKvCache& c, const float* x, const int32_t* steps,
                int64_t T, const DsaScratch& sc, float* out, void* stream, const int32_t* ids_override) {
    if (T < 1 || T > DSA_MAX_T || T > sc.max_t) refuse("decode: 1 <= T <= 8 (and <= the scratch's max_t)");
    if (sc.max_pools < c.max_pools) refuse("decode: scratch sized for fewer pools than the cache");
    const int n = (int) T;
    // 1. the projections of x (one Q8_1 quantization for the four quantized weights)
    native_quantize_q8_1(x, sc.xq, (int) s.n_embd, n, stream);
    native_mmvq(wt.wq_a_type, wt.wq_a, sc.xq, sc.qa, (int) s.n_embd, (int) s.q_lora, n, stream);
    native_mmvq(wt.wkv_a_type, wt.wkv_a, sc.xq, sc.kv, (int) s.n_embd, (int) s.kv_lora, n, stream);
    native_mmvq(wt.idx_k_type, wt.idx_k, sc.xq, sc.ik_raw, (int) s.n_embd, (int) s.idx_dim, n, stream);
    native_mmvq(wt.idx_gate_type, wt.idx_gate, sc.xq, sc.ig, (int) s.n_embd, (int) s.idx_dim, n, stream);
    dsa_gemv_f32(wt.idx_proj, x, sc.iw, s.n_embd, s.idx_n_head, T, 1.0f / sqrtf((float) (s.idx_dim * s.idx_n_head)),
                 stream);
    // 2. the norms
    dsa_rms_norm(sc.qa, wt.q_a_norm, s.rms_eps, s.q_lora, T, sc.qr, stream);
    dsa_rms_norm(sc.kv, wt.kv_a_norm, s.rms_eps, s.kv_lora, T, sc.latent, stream);
    dsa_layer_norm(sc.ik_raw, wt.idx_k_norm, wt.idx_k_norm_b, s.norm_eps, s.idx_dim, T, sc.ik, stream);
    // 3. q and the indexer query from qr
    native_quantize_q8_1(sc.qr, sc.xq, (int) s.q_lora, n, stream);
    native_mmvq(wt.wq_b_type, wt.wq_b, sc.xq, sc.q, (int) s.q_lora, (int) (s.n_head * s.head_k), n, stream);
    native_mmvq(wt.idx_q_b_type, wt.idx_q_b, sc.xq, sc.iq, (int) s.q_lora, (int) (s.idx_n_head * s.idx_dim), n,
                stream);
    // 4. k_b absorbed: q_abs[t][h] = wk_b[h] q[t][h]
    native_quantize_q8_1(sc.q, sc.xq, (int) (T * s.n_head * s.head_k), 1, stream);
    dsa_grouped_q8_0(wt.wk_b, sc.xq, sc.q_abs, s.head_k, s.kv_lora, s.n_head, T, stream);
    // 5. the cache: append, pool, score, select (and make the selection resident when streaming)
    dsa_kv_append(c, s, steps, T, sc.latent, sc.ik, sc.ig, stream);
    dsa_pool_step(c, s, wt.idx_ape, steps, T, stream);
    dsa_index_scores(c, s, sc.iq, sc.iw, steps, T, sc.max_pools, sc.scores, stream);
    dsa_select(sc.scores, s, steps, T, sc.max_pools, sc.cap, sc.ids, stream);
    const int32_t* ids = ids_override ? ids_override : sc.ids;
    dsa_kv_stream_resolve(c, s, ids, steps, T, sc.cap, stream);
    // 6. attention over the latent, then v_b and attn_output
    dsa_attend(c, s, sc.q_abs, ids, steps, T, sc.cap, 1.0f / sqrtf((float) s.head_k), sc.attn, sc.o_lat, stream);
    native_quantize_q8_1(sc.o_lat, sc.xq, (int) (T * s.n_head * s.kv_lora), 1, stream);
    dsa_grouped_q8_0(wt.wv_b, sc.xq, sc.o, s.kv_lora, s.head_v, s.n_head, T, stream);
    native_quantize_q8_1(sc.o, sc.xq, (int) (s.n_head * s.head_v), n, stream);
    native_mmvq(wt.wo_type, wt.wo, sc.xq, out, (int) (s.n_head * s.head_v), (int) s.n_embd, n, stream);
}

}  // namespace strata::kernels
