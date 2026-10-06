// src/prefill/glm_prefill_kernels.cu - see include/strata/prefill/glm_prefill_kernels.hpp.
#include "strata/prefill/glm_prefill_kernels.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::glm {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm prefill kernels: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
[[noreturn]] void refuse(const char* what) {
    std::fprintf(stderr, "glm prefill kernels: %s\n", what);
    std::exit(1);
}

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

// ---- element-wise ---------------------------------------------------------------------------------------------
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f16_from_f32(x[i]);
}

__global__ void gather_f16_kernel(const float* __restrict__ x, int64_t ldx, const int32_t* __restrict__ rows,
                                  int64_t width, uint16_t* __restrict__ out) {
    const int64_t r = blockIdx.y;
    const float* src = x + (int64_t) rows[r] * ldx;
    uint16_t* dst = out + r * width;
    for (int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; k < width; k += (int64_t) gridDim.x * blockDim.x)
        dst[k] = f16_from_f32(src[k]);
}

// ggml-cpu ops.cpp 3614-3616: gate = min(g, limit); up = clamp(u, -limit, limit); gate / (1 + expf(-gate)) * up
__global__ void swiglu_clamp_kernel(const float* __restrict__ gu, int64_t ld, int64_t n_ff, float limit,
                                    uint16_t* __restrict__ h) {
    const int64_t r = blockIdx.y;
    const float* row = gu + r * ld;
    for (int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; k < n_ff; k += (int64_t) gridDim.x * blockDim.x) {
        float g = row[k], u = row[n_ff + k];
        if (limit > 0.0f) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
        }
        const float s = g / (1.0f + expf(-g));
        h[r * n_ff + k] = f16_from_f32(__fmul_rn(s, u));
    }
}

__global__ void scatter_add_kernel(const float* __restrict__ d, const int32_t* __restrict__ tok,
                                   const float* __restrict__ w, int64_t width, float* __restrict__ acc) {
    const int64_t r = blockIdx.y;
    const float wr = w[r];
    float* dst = acc + (int64_t) tok[r] * width;
    const float* src = d + r * width;
    for (int64_t k = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; k < width; k += (int64_t) gridDim.x * blockDim.x)
        dst[k] = __fadd_rn(dst[k], __fmul_rn(src[k], wr));
}

__global__ void add_kernel(const float* a, const float* b, float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __fadd_rn(a[i], b[i]);
}

unsigned row_blocks(int64_t width) { return (unsigned) std::min<int64_t>((width + 255) / 256, 64); }

// ---- the prompt attention ---------------------------------------------------------------------------------------
// grid (n_head / HG, T), 256 threads; warp h of the block is head hg * HG + h.  The token's cells go through in
// stages of CH: a stage's latent rows are staged in shared memory (the int8 codes and their scales, or the f16
// values, as the cache holds them), the HG x CH scores are computed (int8: one thread per (head, cell) and the whole
// dot, the decode kernel's arithmetic; f16: two threads per pair, half the dimensions each), warp h updates head h's
// running max and sum (the online softmax) and leaves the stage's probabilities and the rescale factor in shared
// memory, and every thread folds them into its accumulators: thread t owns dimensions t and t + 256 of all HG heads.
constexpr int HG = 8, THREADS = 256;

template <int FMT, int PER>
__global__ void __launch_bounds__(THREADS) prompt_attend_kernel(strata::kernels::DsaKvCache c,
                                                                const float* __restrict__ q_abs,
                                                                const int32_t* __restrict__ ids, long long cap,
                                                                const int32_t* __restrict__ steps, int n_head,
                                                                int kpool, float scale, float* __restrict__ o) {
    using namespace strata::kernels;
    constexpr int D = PER * 32;
    constexpr bool I8 = FMT == kDsaLatInt8;
    constexpr int CH = I8 ? 32 : 16;                      // cells per stage
    constexpr int G = D / DSA_LAT_GROUP;
    constexpr int DPT = D / THREADS;                      // value dims per thread (1 or 2)
    constexpr int CW = I8 ? D / 4 + 1 : D / 2 + 1;        // 32-bit words per staged row (odd: conflict-free)
    static_assert(D % THREADS == 0 && HG == THREADS / 32, "one warp per head");
    __shared__ __align__(16) float sq[HG][D];
    __shared__ uint32_t sraw[CH][CW];
    __shared__ float sscale[I8 ? CH : 1][I8 ? G + 1 : 1];
    __shared__ float sp[HG][CH];
    __shared__ float spt[CH][HG];
    __shared__ float salpha[HG], sl[HG];
    __shared__ long long srow[CH];
    const int hg = blockIdx.x, tok = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int width = steps[(size_t) tok * kStepCount + kStepWidth];
    const float* qsrc = q_abs + ((size_t) tok * n_head + (size_t) hg * HG) * D;
    for (int i = t; i < HG * D / 4; i += THREADS)
        reinterpret_cast<float4*>(&sq[0][0])[i] = reinterpret_cast<const float4*>(qsrc)[i];
    float acc[HG][DPT];
#pragma unroll
    for (int h = 0; h < HG; ++h)
#pragma unroll
        for (int j = 0; j < DPT; ++j) acc[h][j] = 0.0f;
    float run_m = -INFINITY, run_l = 0.0f;                // warp `warp`'s head: identical in every lane
    for (int c0 = 0; c0 < width; c0 += CH) {
        const int n_here = min(CH, width - c0);
        __syncthreads();                                  // the previous stage is consumed
        if (t < CH) {
            long long r = -1;
            if (t < n_here) {
                const int cell = ids ? ids[(size_t) tok * cap + c0 + t] : c0 + t;
                const int slot = c.page_table[cell / kpool];
                if (slot >= 0) r = (long long) slot * kpool + cell % kpool;
            }
            srow[t] = r;
        }
        __syncthreads();
        // stage the rows: 16-byte loads, CH rows of D bytes (int8) or 2 D bytes (f16)
        constexpr int V16 = I8 ? D / 16 : D / 8;          // 16-byte vectors per row
        for (int i = t; i < CH * V16; i += THREADS) {
            const int cc = i / V16, v = i % V16;
            if (cc < n_here && srow[cc] >= 0) {
                const uint4 w = I8 ? *reinterpret_cast<const uint4*>(c.lat_q + srow[cc] * D + v * 16)
                                   : *reinterpret_cast<const uint4*>(c.lat_f16 + srow[cc] * D + v * 8);
                sraw[cc][v * 4 + 0] = w.x;
                sraw[cc][v * 4 + 1] = w.y;
                sraw[cc][v * 4 + 2] = w.z;
                sraw[cc][v * 4 + 3] = w.w;
            }
        }
        if constexpr (I8) {
            for (int i = t; i < CH * G; i += THREADS) {
                const int cc = i / G, g = i % G;
                if (cc < n_here && srow[cc] >= 0) sscale[cc][g] = f32_from_f16(c.lat_scale[srow[cc] * G + g]);
            }
        }
        __syncthreads();
        // scores
        if constexpr (I8) {
            const int h = warp, cc = lane;
            float s = -INFINITY;
            if (cc < n_here && srow[cc] >= 0) {
                float sacc = 0.0f;
                for (int g = 0; g < G; ++g) {
                    const float sc = sscale[cc][g];
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
                    for (int w = 0; w < DSA_LAT_GROUP / 4; ++w) {
                        const uint32_t word = sraw[cc][g * (DSA_LAT_GROUP / 4) + w];
                        const float4 qv = *reinterpret_cast<const float4*>(&sq[h][g * DSA_LAT_GROUP + 4 * w]);
                        a0 = fmaf((float) (int8_t) (word & 0xffu) * sc, qv.x, a0);
                        a1 = fmaf((float) (int8_t) ((word >> 8) & 0xffu) * sc, qv.y, a1);
                        a2 = fmaf((float) (int8_t) ((word >> 16) & 0xffu) * sc, qv.z, a2);
                        a3 = fmaf((float) (int8_t) (word >> 24) * sc, qv.w, a3);
                    }
                    sacc += (a0 + a1) + (a2 + a3);
                }
                s = sacc * scale;
            }
            sp[h][cc] = s;
        } else {
            const int h = warp, cc = lane & 15, half = lane >> 4;
            const bool live = cc < n_here && srow[cc] >= 0;
            float a0 = 0.0f, a1 = 0.0f;
            if (live) {
                const int w0 = half * (D / 4);            // D / 2 values = D / 4 words per half
                for (int w = 0; w < D / 4; ++w) {
                    const uint32_t word = sraw[cc][w0 + w];
                    const int d = 2 * (w0 + w);
                    a0 = fmaf(f32_from_f16((uint16_t) (word & 0xffffu)), sq[h][d], a0);
                    a1 = fmaf(f32_from_f16((uint16_t) (word >> 16)), sq[h][d + 1], a1);
                }
            }
            float s = a0 + a1;
            s += __shfl_xor_sync(0xffffffffu, s, 16);
            if (half == 0) sp[h][cc] = live ? s * scale : -INFINITY;
        }
        __syncthreads();
        // the online softmax: warp h updates head h
        {
            const int h = warp;
            const float v = lane < CH ? sp[h][lane] : -INFINITY;
            float m = v;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
            const float new_m = fmaxf(run_m, m);
            const float alpha = new_m == -INFINITY ? 1.0f : expf(run_m - new_m);
            const float p = (lane < CH && v != -INFINITY) ? expf(v - new_m) : 0.0f;
            float ps = p;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
            run_l = run_l * alpha + ps;
            run_m = new_m;
            if (lane < CH) spt[lane][h] = p;
            if (lane == 0) salpha[h] = alpha;
        }
        __syncthreads();
        // p . V
        float al[HG];
#pragma unroll
        for (int h = 0; h < HG; ++h) al[h] = salpha[h];
#pragma unroll
        for (int h = 0; h < HG; ++h)
#pragma unroll
            for (int j = 0; j < DPT; ++j) acc[h][j] *= al[h];
        for (int cc = 0; cc < n_here; ++cc) {
            if (srow[cc] < 0) continue;
            const float4 p0 = *reinterpret_cast<const float4*>(&spt[cc][0]);
            const float4 p1 = *reinterpret_cast<const float4*>(&spt[cc][4]);
            const float p[HG] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int j = 0; j < DPT; ++j) {
                const int d = t + j * THREADS;
                float v;
                if constexpr (I8)
                    v = (float) (int8_t) ((sraw[cc][d >> 2] >> (8 * (d & 3))) & 0xffu) * sscale[cc][d / DSA_LAT_GROUP];
                else
                    v = f32_from_f16((uint16_t) ((sraw[cc][d >> 1] >> (16 * (d & 1))) & 0xffffu));
#pragma unroll
                for (int h = 0; h < HG; ++h) acc[h][j] = fmaf(p[h], v, acc[h][j]);
            }
        }
    }
    if (lane == 0) sl[warp] = run_l;
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HG; ++h) {
        const float l = sl[h];
        float* dst = o + ((size_t) tok * n_head + (size_t) hg * HG + h) * D;
#pragma unroll
        for (int j = 0; j < DPT; ++j) dst[t + j * THREADS] = l > 0.0f ? acc[h][j] / l : 0.0f;
    }
}

}  // namespace

// ---- Gemm16 -----------------------------------------------------------------------------------------------------
Gemm16::~Gemm16() {
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
}

bool Gemm16::init(void* stream, std::string& err, void* workspace, size_t ws_bytes) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "glm prefill gemm: cublasCreate: status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    if (cublasSetStream(h, (cudaStream_t) stream) != CUBLAS_STATUS_SUCCESS) {
        err = "glm prefill gemm: cublasSetStream failed";
        return false;
    }
    if (workspace != nullptr && ws_bytes > 0) cublasSetWorkspace(h, workspace, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    return true;
}

void Gemm16::run(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t ldy, int64_t T, int64_t N,
                 int64_t K, float beta) {
    if (T <= 0 || N <= 0) return;
    const float alpha = 1.0f;
    // column-major view: Y^T[N, T] = W[N, K] . X^T[K, T]; W is row-major [N][K] = column-major [K][N] transposed
    const cublasStatus_t s = cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,
                                          &alpha, W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) ldx, &beta, Y,
                                          CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "glm prefill gemm: GemmEx T=%lld N=%lld K=%lld: status %d\n", (long long) T, (long long) N,
                     (long long) K, (int) s);
        std::exit(1);
    }
}

// ---- entry points -----------------------------------------------------------------------------------------------
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}

void gather_f16(const float* x, int64_t ldx, const int32_t* rows, int64_t n, int64_t width, uint16_t* out,
                void* stream) {
    if (n <= 0) return;
    if (n > 65535) refuse("gather_f16: more than 65535 rows");
    gather_f16_kernel<<<dim3(row_blocks(width), (unsigned) n), 256, 0, (cudaStream_t) stream>>>(x, ldx, rows, width,
                                                                                               out);
    check("gather_f16");
}

void swiglu_clamp_f16(const float* gu, int64_t ld, int64_t rows, int64_t n_ff, float limit, uint16_t* h,
                      void* stream) {
    if (rows <= 0) return;
    if (rows > 65535) refuse("swiglu_clamp_f16: more than 65535 rows");
    swiglu_clamp_kernel<<<dim3(row_blocks(n_ff), (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(gu, ld, n_ff,
                                                                                                    limit, h);
    check("swiglu_clamp_f16");
}

void scatter_add(const float* d, const int32_t* tok, const float* w, int64_t n, int64_t width, float* acc,
                 void* stream) {
    if (n <= 0) return;
    if (n > 65535) refuse("scatter_add: more than 65535 rows");
    scatter_add_kernel<<<dim3(row_blocks(width), (unsigned) n), 256, 0, (cudaStream_t) stream>>>(d, tok, w, width, acc);
    check("scatter_add");
}

void add(const float* a, const float* b, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(a, b, y, n);
    check("add");
}

void expert_rows(Gemm16& gemm, const strata::kernels::NativeExpertLayout& L, const uint8_t* blob, const float* x,
                 int64_t ldx, const int32_t* rows, const float* w, int64_t n, float limit, const ExpertScratch& s,
                 float* out, void* stream) {
    if (n <= 0) return;
    const int64_t E = L.n_embd, F = L.n_ff;
    if (L.up_off != (size_t) F * L.gu_row) refuse("expert_rows: gate and up rows are not contiguous");
    gather_f16(x, ldx, rows, n, E, s.x16, stream);
    strata::kernels::iq_dequant_f16(L.gu_type, blob, 2 * F * E, s.w16, stream);
    gemm.run(s.x16, E, s.w16, s.gu, 2 * F, n, 2 * F, E);
    swiglu_clamp_f16(s.gu, 2 * F, n, F, limit, s.h16, stream);
    strata::kernels::iq_dequant_f16(L.d_type, blob + L.down_off, E * F, s.w16, stream);
    gemm.run(s.h16, F, s.w16, s.d, E, n, E, F);
    scatter_add(s.d, rows, w, n, E, out, stream);
}

void dsa_prompt_attend(const strata::kernels::DsaKvCache& c, const strata::kernels::DsaShapes& s, const float* q_abs,
                       const int32_t* ids, int64_t cap, const int32_t* steps, int64_t T, float scale, float* o,
                       void* stream) {
    using namespace strata::kernels;
    if (T <= 0) return;
    if (s.n_head % HG != 0) refuse("dsa_prompt_attend: n_head must be a multiple of 8");
    if (s.kv_lora != 512 && s.kv_lora != 256) refuse("dsa_prompt_attend: kv_lora must be 256 or 512");
    if (T > 65535) refuse("dsa_prompt_attend: more than 65535 tokens");
    if (c.n_slots < c.n_pages) refuse("dsa_prompt_attend: the latent cache must be fully resident");
    const dim3 grid((unsigned) (s.n_head / HG), (unsigned) T);
    cudaStream_t st = (cudaStream_t) stream;
    const int H = (int) s.n_head, K = (int) s.kpool;
    if (c.fmt == kDsaLatInt8) {
        if (s.kv_lora == 512)
            prompt_attend_kernel<kDsaLatInt8, 16><<<grid, THREADS, 0, st>>>(c, q_abs, ids, cap, steps, H, K, scale, o);
        else
            prompt_attend_kernel<kDsaLatInt8, 8><<<grid, THREADS, 0, st>>>(c, q_abs, ids, cap, steps, H, K, scale, o);
    } else {
        if (s.kv_lora == 512)
            prompt_attend_kernel<kDsaLatF16, 16><<<grid, THREADS, 0, st>>>(c, q_abs, ids, cap, steps, H, K, scale, o);
        else
            prompt_attend_kernel<kDsaLatF16, 8><<<grid, THREADS, 0, st>>>(c, q_abs, ids, cap, steps, H, K, scale, o);
    }
    check("dsa_prompt_attend");
}

}  // namespace strata::prefill::glm
