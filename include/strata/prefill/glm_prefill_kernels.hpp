// include/strata/prefill/glm_prefill_kernels.hpp - GLM-5.3-Flash (llama.cpp arch `glm5-next`) prompt path: the
// kernels a chunk of many tokens needs beside the decode kernels it reuses.
//
// The prompt path (include/strata/prefill/glm_prefill.hpp) runs every projection as an FP16 tensor-core GEMM (the
// GGUF blocks dequantized to FP16, the activations rounded to FP16, F32 accumulation and output: what Strata's Qwen
// prompt path does with `Gemm::native`) and reuses the decode kernels that already take any number of tokens (mHC,
// the norms, the router, the DSA append / pooling / scores / selection, the KDA conv and recurrence through
// kda_conv_l2_prompt / kda_step_norm_prompt).  What is new is here:
//
//   Gemm16              Y[T, N] (f32) = X[T, K] (f16, row stride ldx) . W[N, K]^T (f16 row-major), cuBLAS / hipBLAS
//   to_f16, gather_f16  activations to FP16 (all rows, or the rows of one expert's entries)
//   swiglu_clamp_f16    h = f16(silu(min(gate, limit)) * clamp(up, -limit, limit)), ggml_swiglu_clamp's arithmetic
//                       (ggml-cpu ops.cpp 3614-3616), from one buffer holding gate | up per row
//   scatter_add         acc[tok[i]] += w[i] * d[i]: an expert's rows into the MoE output (each token once per call)
//   add                 y = a + b
//   dsa_prompt_attend   the sparse MLA attention of many tokens, each over its own selection (or every cell up to
//                       it), with an online softmax: one block per (token, 8 heads), the selected latent rows staged
//                       in shared memory once per block and used by its 8 heads
//
// All pointers are device memory, every call enqueues on `stream` and synchronizes nothing.
#pragma once

#include "strata/kernels/dsa.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::prefill::glm {

/// The FP16 GEMM of the prompt path (one handle, the caller's stream).  CUDA: cublasGemmEx (tensor cores).  HIP with
/// hipBLASLt: hipblasLtMatmul with the library's first heuristic algorithm for the shape, cached per shape (measured
/// on the RX 9070 XT: 60-130 TFLOPS against hipblasGemmEx's ~11 for these f16 x f16 -> f32 products), hipblasGemmEx
/// when hipBLASLt has none.  STRATA_GLM_GEMM_LT=0 keeps hipblasGemmEx.
class Gemm16 {
public:
    Gemm16() = default;
    ~Gemm16();
    Gemm16(const Gemm16&) = delete;
    Gemm16& operator=(const Gemm16&) = delete;
    /// `workspace` (device, `ws_bytes`) may be null: the library then uses its own.
    bool init(void* stream, std::string& err, void* workspace = nullptr, size_t ws_bytes = 0);
    /// Y[t][n] (row stride ldy) = beta * Y + sum_k X[t][k] (row stride ldx) * W[n][k] (row stride K).
    void run(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t ldy, int64_t T, int64_t N, int64_t K,
             float beta = 0.0f);
    void* stream() const { return stream_; }

private:
    void* handle_ = nullptr;
    void* stream_ = nullptr;
    void* lt_ = nullptr;   ///< HIP: the hipBLASLt state (handle, workspace, the per-shape plans)
};

/// y[i] = f16(x[i]) (round to nearest even), n values.
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
/// out[i][k] = f16(x[rows[i]][k]) for k < width; x has `ldx` floats per row, out `width` halves per row.
void gather_f16(const float* x, int64_t ldx, const int32_t* rows, int64_t n, int64_t width, uint16_t* out,
                void* stream);
/// h[r][k] = f16(g / (1 + expf(-g)) * u), g = min(gu[r][k], limit), u = clamp(gu[r][n_ff + k], -limit, limit)
/// (limit <= 0: no clamping, the plain SwiGLU).  gu has `ld` floats per row, h n_ff halves per row.
void swiglu_clamp_f16(const float* gu, int64_t ld, int64_t rows, int64_t n_ff, float limit, uint16_t* h,
                      void* stream);
/// acc[tok[i]][k] = acc[tok[i]][k] + w[i] * d[i][k] (the product and the sum each rounded, as ggml_mul then
/// ggml_add); every tok[i] of one call distinct.  d and acc have `width` floats per row.
void scatter_add(const float* d, const int32_t* tok, const float* w, int64_t n, int64_t width, float* acc,
                 void* stream);
/// y[i] = a[i] + b[i] (y may alias a).
void add(const float* a, const float* b, float* y, int64_t n, void* stream);

/// One routed expert over its rows of a prompt chunk (the MoE's grouped GEMM): for i < n, with r = rows[i],
///   out[r] = out[r] + w[i] * down(swiglu_clamp(gate x[r], up x[r], limit))
/// The expert is a native blob (L: gate rows | up rows | down rows, GGUF blocks; IQ3_S / IQ4_XS gate/up, IQ4_XS /
/// Q6_K down and every type iq_dequant_f16 takes) in device memory.  Its gate|up and down are dequantized to FP16 in
/// `w16` (2 n_ff n_embd halves; gate|up first, then down over it), x's rows gathered to FP16, gate|up = one GEMM
/// [n][2 n_ff], the clamped SwiGLU to FP16, down = one GEMM [n][n_embd] (F32), then the weighted rows added
/// (scatter_add: n <= 65535, every rows[i] distinct).  rows and w are device arrays.
struct ExpertScratch {
    uint16_t* w16 = nullptr;   ///< 2 n_ff n_embd halves
    uint16_t* x16 = nullptr;   ///< n n_embd halves
    float* gu = nullptr;       ///< n 2 n_ff floats
    uint16_t* h16 = nullptr;   ///< n n_ff halves
    float* d = nullptr;        ///< n n_embd floats
};
void expert_rows(Gemm16& gemm, const strata::kernels::NativeExpertLayout& L, const uint8_t* blob, const float* x,
                 int64_t ldx, const int32_t* rows, const float* w, int64_t n, float limit, const ExpertScratch& s,
                 float* out, void* stream);

/// The sparse MLA attention of T tokens (a prompt chunk's queries).  Token t attends to steps[t].width cells: the
/// ids row t (`ids` [T][cap], from dsa_select), or, with ids == null, cells 0 .. width - 1 (the dense regime).
/// p = softmax(q_abs[t][h] . latent[cell] * scale) over those cells, o[t][h] = sum p latent[cell] (K = V = the
/// cached latent, int8 or f16 as the cache holds it).  q_abs, o: [T][n_head][kv_lora] f32.  Requires a fully
/// resident cache (the prompt path does not stream the latent) and n_head a multiple of 8, kv_lora 512 or 256.
void dsa_prompt_attend(const strata::kernels::DsaKvCache& c, const strata::kernels::DsaShapes& s, const float* q_abs,
                       const int32_t* ids, int64_t cap, const int32_t* steps, int64_t T, float scale, float* o,
                       void* stream);

}  // namespace strata::prefill::glm
