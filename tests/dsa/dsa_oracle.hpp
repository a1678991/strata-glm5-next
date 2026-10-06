// tests/dsa/dsa_oracle.hpp - the CPU oracle for the DSA kernels: ggml at llama.cpp f1cee99, CPU backend, the ops of
// `build_dsa_layer` / `build_kpool_select` (src/models/glm5-next.cpp:766-1013) called directly, in the same order,
// for one sequence whose cells are its positions 0..N-1. The k-pool inputs (pool_cells, pool_idxs, pool_mask,
// tail_idxs, gather_mask) are filled on the host exactly as `set_input_kpool` (src/llama-memory-hybrid-idx.cpp
// :791-1036) fills them for such a sequence, and the attention is llama.cpp's gather path (G:965-990).
//
// The cache is host state shared by the phases: the latent as the cache holds it (f16, or the int8 codes
// dequantized - the oracle quantizes its own latent with the engine's int8 rule, so it sees what an int8 cache
// would hold), and the indexer rows key | gate and the pooled rows in f16, as llama.cpp's idx cache holds them.
#pragma once

#include "strata/kernels/dsa.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace dsa_test {

using strata::kernels::DsaShapes;

/// Random layer weights. Matrices are Q8_0 GGUF bytes quantized by ggml (the GPU gets the same bytes).
struct HostWeights {
    std::vector<uint8_t> wq_a, wq_b, wkv_a, wk_b, wv_b, wo, idx_q_b, idx_k, idx_gate;
    std::vector<float> q_a_norm, kv_a_norm, idx_k_norm, idx_k_norm_b, idx_ape, idx_proj;
};
HostWeights make_weights(const DsaShapes& s, uint64_t seed);

/// The engine's int8 latent rule (kv_q8.hpp), on the host: codes and f16 scale bits per 64 values.
void quantize_latent_row(const float* x, int n, int8_t* codes, uint16_t* scales);
float f16_round(float x);

/// The oracle's cache.
struct HostCache {
    int fmt = strata::kernels::kDsaLatInt8;
    int64_t n_cells = 0;              ///< cells written so far (positions 0..n_cells-1)
    std::vector<float> latent;        ///< [max_cells][kv_lora] as the cache holds it (dequantized / f16-rounded)
    std::vector<int8_t> codes;        ///< int8: [max_cells][kv_lora]
    std::vector<uint16_t> scales;     ///< int8: [max_cells][kv_lora / 64]
    std::vector<uint16_t> kg;         ///< [max_cells][2 * idx_dim] f16
    std::vector<uint16_t> pooled;     ///< [max_pools][idx_dim] f16
};
void cache_init(HostCache& c, const DsaShapes& s, int fmt, int64_t max_cells);
/// Writes cells p0.. (latent f32 rows, key, gate) into the cache the way the engine's append does.
void cache_append(HostCache& c, const DsaShapes& s, int64_t p0, int64_t n, const float* latent, const float* key,
                  const float* gate);
/// Pools [j0, j1) with ggml (G:801-829): get_rows of key | gate, add ape, soft_max over the members, mul, sum_rows,
/// then the f16 store.
void oracle_pool(HostCache& c, const DsaShapes& s, const std::vector<float>& ape, int64_t j0, int64_t j1,
                 int n_threads);

/// One step's values, per token t (row-major [T][...]).
struct StepOut {
    std::vector<float> qa, kv, ik_raw;                         ///< phase A, before the norms
    std::vector<float> qr, q, q_abs, latent, ik, ig, iw, iq;   ///< phase A
    std::vector<float> scores;                                 ///< [T][n_pool]: the indexer score + pool mask
    int64_t n_pool = 0;                                        ///< llama.cpp's padded pool count
    std::vector<std::vector<int32_t>> sel;                     ///< per token: the live selected cells, ascending
    std::vector<std::vector<int32_t>> slots;                   ///< per token: llama.cpp's slot order, -1 = dead
    std::vector<float> o_lat, o, out;                          ///< phase D
};

/// Phase A: the projections and norms for x [T][n_embd].
void oracle_project(const HostWeights& w, const DsaShapes& s, const std::vector<float>& x, int64_t T, StepOut& r,
                    int n_threads);
/// Phase C: lightning indexer, top_k, argsort, the pool -> cell expansion and the tail, for tokens at p0..p0+T-1
/// against the cache's pools (the cache must already hold cells up to p0+T-1 and their pools).
void oracle_select(const HostCache& c, const DsaShapes& s, const StepOut& a, int64_t p0, int64_t T, StepOut& r,
                   int n_threads);
/// Phase D: the gather-path attention over `slots` (per token: cells, -1 = a dead slot, masked), v_b and
/// attn_output.
void oracle_attend(const HostWeights& w, const HostCache& c, const DsaShapes& s, const StepOut& a, int64_t p0,
                   int64_t T, const std::vector<std::vector<int32_t>>& slots, StepOut& r, int n_threads);

/// Max abs and max relative error (relative to the reference's max |value|) between two arrays.
struct Err {
    double max_abs = 0.0, ref_max = 0.0, rel = 0.0;
    int64_t n = 0;
};
Err compare(const float* got, const float* ref, int64_t n);

/// Activation quantization. A quantized product on the GPU quantizes its activation to Q8_1 (native_quantize_q8_1:
/// d = max|x| / 127, code = roundf(x / d)), ggml-cpu on x86 to Q8_0 (code = rint(x * (127 / max|x|))); both store d
/// as f16. The codes agree except near a rounding tie, and there the GPU's own arithmetic decides: CUDA builds the
/// quantizer with --use_fast_math (llama.cpp CUDA's contract), so both divisions are div.approx.ftz.f32 (2 ulp) and a
/// quotient within a few ulp of a tie can round either way; HIP divides in IEEE. So the bound is taken from the
/// GPU's own blocks, read back after the product: for x [cols][n_in] (32-value blocks) and the GPU's Q8_1 blocks of
/// it, `steps` is the largest per-column sum over the block values of |d_gpu * code_gpu - d_cpu * code_cpu| (f16
/// scales), and the column of the product then differs by at most steps * max |w| beyond f32 rounding.
struct FlipBound {
    double steps = 0.0;
    int64_t flips = 0;      ///< block values whose GPU and CPU dequantized values differ
    int64_t off_rule = 0;   ///< GPU codes off the host's roundf(x / d), each one step at a near-tie (CUDA)
    int64_t bad = 0;        ///< GPU codes or scales no rounding of the rule explains (a broken quantizer or readback)
};
FlipBound q8_1_flip_bound(const float* x, const void* gpu_q8_1, int64_t cols, int64_t n_in);
/// max |w| of a Q8_0 matrix (dequantized).
double q8_0_max_abs(const std::vector<uint8_t>& w);

}  // namespace dsa_test
