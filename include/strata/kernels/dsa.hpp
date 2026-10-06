// include/strata/kernels/dsa.hpp - GLM-5.3-Flash (llama.cpp arch `glm5-next`): the DSA layer's decode-step kernels.
//
// Port plan (~/dev/llm/docs/strata-glm-port.md, Stage 4, worker D). The reference is llama.cpp f1cee99:
// `build_dsa_layer` (src/models/glm5-next.cpp:923-1013), `build_kpool_select` (:766-919), the k-pool inputs
// (`set_input_kpool`, src/llama-memory-hybrid-idx.cpp:791-1036) and the CPU ops they call. One DSA layer, one
// sequence, a decode step of T <= 8 tokens at positions p0..p0+T-1 (one token, or a verify window):
//
//   qr     = rms_norm(wq_a x) * q_a_norm                      [q_lora]          (eps attention.layer_norm_rms_epsilon)
//   q      = wq_b qr                                          [n_head][head_k]  (no RoPE: G:935 asserts nope-only)
//   latent = rms_norm(wkv_a x) * kv_a_norm                    [kv_lora]         -> the KV cache, K = V = latent
//   q_abs  = wk_b[h] q[h]                                     [n_head][kv_lora] (k_b absorbed into q)
//   iq     = idx_q_b qr                                       [idx_n_head][idx_dim]
//   ik     = norm(idx_k x) * k_norm + k_norm_b                [idx_dim]         (eps attention.layer_norm_epsilon)
//   ig     = idx_gate x                                       [idx_dim]
//   w      = idx_proj x / sqrt(idx_dim * idx_n_head)          [idx_n_head]      (F32 weight)
//   pooled[j] (pool j = cells 4j..4j+3, written when its last cell arrives):
//            pooled[c] = sum_k softmax_k(ig_k[c] + ape[k][c]) * ik_k[c]                    (f16 in the cache)
//   score[j] = sum_h relu(iq_h . pooled[j]) * w_h   over the pools that END at or before the token
//   cells  = the top idx_top_k/kpool (512) pools' cells + the token's incomplete tail ((pos+1) % kpool cells)
//            = every cell 0..pos while pos + 1 <= 2051: the selection is the identity there
//   o_lat  = softmax(q_abs . latent[cells] / sqrt(head_k)) . latent[cells]            [n_head][kv_lora]
//   out    = wo (wv_b[h] o_lat[h])                            [n_embd]
//
// Every dimension is a runtime field of DsaShapes. Positions, cache sizes and selection widths reach the kernels
// through a device step array ([T][kStepCount], qsa.hpp's QsaStep layout: pos, n_kv = pos + 1, n_bid = the pools
// complete at pos, width = the cells the token attends to), and every grid is sized by a capacity, so a step is
// capturable in a CUDA graph exactly as the QSA layer is.
//
// A window token attends to what llama.cpp's mask gives it: its own selection, computed from its own query, over
// the pools that end at or before its position, plus its own tail. While the selection is the identity this is
// plain causal attention within the window; once it is sparse a window token sees an earlier window token only
// through its tail or a selected pool, exactly as llama.cpp's gather/scatter masks (G:889-918) do.
//
// THE CACHE (one DSA layer, one sequence). A page is `page_size` = kpool = 4 cells = one indexer pool:
//
//   int8 latent (production):  lat_q     int8 [slot][page_size][kv_lora]            codes
//                              lat_scale f16  [slot][page_size][kv_lora / 64]       one scale per 64 values
//                              kg        f16  [slot][page_size][2 * idx_dim]        key | gate, as llama.cpp's row
//                              pad       f16  [slot][page_size][kv_lora / 64]       unused (see streaming)
//   f16 latent (parity):       lat_f16   f16  [slot][page_size][kv_lora]
//                              kg        as above
//   both:                      pooled    f16  [max_pools][idx_dim]   always resident: every step scores every pool
//                              page_table i32 [n_pages]               logical page -> slot, -1 = not resident
//
// The int8 scheme is kv_q8.hpp's: scale = f16(max|x| / 127) per 64 values, code = clamp(rint(x / scale), +-127).
// Bytes per cell and layer: int8 512 + 16 + 512 + 16 + 64 (pooled, one row per 4 cells) = 1,120;
// f16 1,024 + 512 + 64 = 1,600; llama.cpp f16: 1,024 (latent) + 768 (key | gate | pooled per cell) = 1,792.
//
// STREAMING (--kv-resident). With a host copy the authoritative pages live in pinned, device-mapped memory in the
// identity layout and VRAM holds `n_slots` pages; `page_table` is the residency map of kv_stream.hpp. The int8 page
// is laid out so that kv_stream.cu's int8 runs carry it unchanged: viewed as a QSA layer with n_head_kv = 1,
// head_dim = kv_lora and page_size = kpool, its K codes are lat_q, its K scales lat_scale, its V "codes" the
// key | gate rows and its V scales the pad. A V code run is kv_lora bytes per cell and a key | gate row is
// 2 * idx_dim f16 = 4 * idx_dim bytes, so the view needs 4 * idx_dim == kv_lora (GLM: 4 x 128 = 512);
// `dsa_kv_stream_resolve` refuses any other geometry, and the f16 latent (a parity mode) is never streamed.
// Writers write the host copy always and the slot only if resident; the pooling reads a non-resident page's
// key | gate rows from the host copy, so no page has to be made resident just to complete a pool.
#pragma once

#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// One DSA layer's geometry. GLM-5.3-Flash: dsa_glm5_shapes().
struct DsaShapes {
    int64_t n_embd = 0;      ///< 4096
    int64_t n_head = 0;      ///< 64 query heads
    int64_t q_lora = 0;      ///< 1536
    int64_t kv_lora = 0;     ///< 512: the latent, K = V
    int64_t head_k = 0;      ///< 256: key_length_mla, the per-head nope dim of q (no rope part: n_rot = 0)
    int64_t head_v = 0;      ///< 256: value_length_mla, the per-head output of v_b
    int64_t idx_n_head = 0;  ///< 32 indexer heads
    int64_t idx_dim = 0;     ///< 128
    int64_t kpool = 0;       ///< 4 cells per pool (= page_size)
    int64_t idx_top_k = 0;   ///< 2048 cells of selection budget = 512 pools
    float rms_eps = 0.0f;    ///< attention.layer_norm_rms_epsilon (q_a_norm, kv_a_norm): 1e-5
    float norm_eps = 0.0f;   ///< attention.layer_norm_epsilon (the indexer key LayerNorm): 1e-6
};

/// GLM-5.3-Flash's DSA geometry (GGUF header 2026-10-06; glm5-next.* keys). The eps values are the GGUF's float32
/// values as llama.cpp reads them.
inline DsaShapes dsa_glm5_shapes() {
    DsaShapes s;
    s.n_embd = 4096; s.n_head = 64; s.q_lora = 1536; s.kv_lora = 512; s.head_k = 256; s.head_v = 256;
    s.idx_n_head = 32; s.idx_dim = 128; s.kpool = 4; s.idx_top_k = 2048;
    s.rms_eps = 9.999999747378752e-06f;
    s.norm_eps = 9.999999974752427e-07f;
    return s;
}

inline constexpr int DSA_LAT_GROUP = 64;   ///< int8 latent: one f16 scale per 64 values (kv_q8.hpp's group)
inline constexpr int DSA_MAX_T = 8;        ///< tokens per step (Strata's kVerifyMaxT)

/// Pools the selection keeps: idx_top_k / kpool (512).
inline int64_t dsa_top_pools(const DsaShapes& s) { return s.idx_top_k / s.kpool; }
/// The largest selection: top pools plus a tail of kpool - 1 cells (2,051). The capacity of an id row.
inline int64_t dsa_sel_cap(const DsaShapes& s) { return dsa_top_pools(s) * s.kpool + s.kpool - 1; }
/// The cells the token at `pos` attends to: min(complete pools, 512) * kpool + its tail.
inline int64_t dsa_sel_width(int64_t pos, const DsaShapes& s) {
    const int64_t n_bid = (pos + 1) / s.kpool, top = dsa_top_pools(s);
    return (n_bid < top ? n_bid : top) * s.kpool + (pos + 1) % s.kpool;
}
/// Pool rows the cache needs for `max_cells` cells.
inline int64_t dsa_max_pools(int64_t max_cells, const DsaShapes& s) { return max_cells / s.kpool + 1; }

/// Fills `T` HOST step records ([T][kStepCount]) for positions p0..p0+T-1: pos, n_kv = pos + 1,
/// n_bid = (pos + 1) / kpool, width = dsa_sel_width(pos). Upload them with one copy.
void dsa_step_fill(int32_t* host_steps, int64_t p0, int64_t T, const DsaShapes& s);

// ================================ the latent KV cache ================================

enum DsaLatentFmt : int { kDsaLatInt8 = 0, kDsaLatF16 = 1 };

/// The authoritative copy of a streamed layer: device-mapped pointers into pinned host memory, identity layout
/// (page p in slot p). All null when the layer is fully resident.
struct DsaKvHost {
    int8_t* lat_q = nullptr;
    uint16_t* lat_scale = nullptr;
    uint16_t* lat_f16 = nullptr;
    uint16_t* kg = nullptr;
    uint16_t* pad = nullptr;
    bool present() const { return lat_q != nullptr || lat_f16 != nullptr; }
};

/// One layer's cache. Every pointer is device memory (the host copy device-mapped).
struct DsaKvCache {
    int fmt = kDsaLatInt8;
    int8_t* lat_q = nullptr;        ///< int8: [slot][page_size][kv_lora]
    uint16_t* lat_scale = nullptr;  ///< int8: [slot][page_size][kv_lora / 64]
    uint16_t* lat_f16 = nullptr;    ///< f16:  [slot][page_size][kv_lora]
    uint16_t* kg = nullptr;         ///< [slot][page_size][2 * idx_dim] key | gate
    uint16_t* pad = nullptr;        ///< int8: [slot][page_size][kv_lora / 64], unused (streaming view)
    uint16_t* pooled = nullptr;     ///< [max_pools][idx_dim]
    int32_t* page_table = nullptr;  ///< [n_pages]
    int64_t n_pages = 0, n_slots = 0, max_cells = 0, max_pools = 0;
    DsaKvHost host;                 ///< streaming only
    KvStreamMap map;                ///< streaming only: page_table, slot arrays, ctl (kv_stream.hpp)
};

/// Bytes of one page (slot) of latent + key | gate (+ pad).
uint64_t dsa_kv_page_bytes(const DsaShapes& s, int fmt);
/// Bytes per cell of one layer, the pooled row's share (idx_dim * 2 / kpool) included.
uint64_t dsa_kv_cell_bytes(const DsaShapes& s, int fmt);
/// The VRAM arena of one layer: `n_slots` pages (n_slots >= n_pages: fully resident), the page table, the pooled
/// rows, and with n_slots < n_pages the streaming map. Every region 256-byte aligned.
uint64_t dsa_kv_bytes(const DsaShapes& s, int fmt, int64_t max_cells, int64_t n_slots);
/// Lays the arena out over `base` (device memory of dsa_kv_bytes) and returns the bytes used. Does not clear it.
uint64_t dsa_kv_carve(void* base, const DsaShapes& s, int fmt, int64_t max_cells, int64_t n_slots, DsaKvCache& c);
/// The host copy of a streamed layer: n_pages pages in the identity layout.
uint64_t dsa_kv_host_bytes(const DsaShapes& s, int fmt, int64_t max_cells);
/// Lays the host copy out over `dev_base` (the DEVICE pointer of pinned, mapped memory) into c.host.
void dsa_kv_host_carve(void* dev_base, const DsaShapes& s, int fmt, int64_t max_cells, DsaKvCache& c);
/// Clears the slots and pooled rows; the page table becomes the identity (fully resident) or all -1 with an empty
/// streaming map. Not capturable (memset).
void dsa_kv_reset(DsaKvCache& c, const DsaShapes& s, void* stream);

/// Appends the T cells of a step at steps[i].pos: the latent [T][kv_lora] (quantized to int8 or rounded to f16),
/// and the indexer key and gate [T][idx_dim] as f16. Writes the host copy when present and the slot when the page
/// is resident. Capturable.
void dsa_kv_append(const DsaKvCache& c, const DsaShapes& s, const int32_t* steps, int64_t T, const float* latent,
                   const float* key, const float* gate, void* stream);
/// The same for n cells at host positions p0..p0+n-1 (a prompt chunk, a test fixture). Not for a captured graph.
void dsa_kv_append_range(const DsaKvCache& c, const DsaShapes& s, int64_t p0, int64_t n, const float* latent,
                         const float* key, const float* gate, void* stream);
/// Reads logical page `page` (from its slot, or from the host copy when it is not resident): the latent
/// dequantized [page_size][kv_lora] f32 and key | gate [page_size][2 * idx_dim] f32 (either may be null).
void dsa_kv_read_page(const DsaKvCache& c, const DsaShapes& s, int64_t page, float* latent, float* kg, void* stream);

/// Streaming: make every page named by the T selections resident (kv_stream_resolve over the int8 page viewed as
/// a QSA int8 page, see above). ids [T][cap], widths from steps. Capturable. No-op for a fully resident cache.
void dsa_kv_stream_resolve(const DsaKvCache& c, const DsaShapes& s, const int32_t* ids, const int32_t* steps,
                           int64_t T, int64_t cap, void* stream);

// ================================ the indexer ================================

/// Pools completed by the step (pool j completes at position kpool*j + kpool - 1, within steps[0].pos ..
/// steps[T-1].pos): pooled[j] = f16(sum_k p_k * key_k), p = softmax_k(gate_k + ape[k]) per channel, with
/// llama.cpp's CPU arithmetic (softmax in f32 with a double sum, then a double row sum). `ape` is
/// indexer_compressor_ape, F32 [kpool][idx_dim]. Capturable.
void dsa_pool_step(const DsaKvCache& c, const DsaShapes& s, const float* ape, const int32_t* steps, int64_t T,
                   void* stream);
/// The same for pools [j0, j1) (host indices: a prompt chunk, a fixture). Not for a captured graph.
void dsa_pool_range(const DsaKvCache& c, const DsaShapes& s, const float* ape, int64_t j0, int64_t j1, void* stream);

/// Per-pool lightning-indexer scores of T tokens: scores[t][j] = sum_h relu(iq[t][h] . pooled[j]) * w[t][h] for
/// j < steps[t].n_bid (the pools that end at or before the token); the rest of the row is not written.
/// iq [T][idx_n_head][idx_dim] f32, w [T][idx_n_head] f32 (already scaled), scores [T][max_pools] with the row
/// stride max_pools >= c.max_pools (the same stride dsa_select reads). Capturable.
void dsa_index_scores(const DsaKvCache& c, const DsaShapes& s, const float* iq, const float* w, const int32_t* steps,
                      int64_t T, int64_t max_pools, float* scores, void* stream);

/// The selection of T tokens: ids[t][0 .. width) = the cells of the top min(n_bid, 512) pools by score (ties to
/// the lower pool), ascending, then the token's tail cells pos - n_tail + 1 .. pos. ids [T][cap], cap >=
/// dsa_sel_cap. Capturable.
void dsa_select(const float* scores, const DsaShapes& s, const int32_t* steps, int64_t T, int64_t max_pools,
                int64_t cap, int32_t* ids, void* stream);

// ================================ MLA ================================

/// y[r] = rms_norm(x[r]) * w, rows of n; ggml-cpu's arithmetic (double sum of squares, f32 scale).
void dsa_rms_norm(const float* x, const float* w, float eps, int64_t n, int64_t rows, float* y, void* stream);
/// y[r] = norm(x[r]) * w + b (ggml_norm: double sums, f32 scale).
void dsa_layer_norm(const float* x, const float* w, const float* b, float eps, int64_t n, int64_t rows, float* y,
                    void* stream);
/// y[col][o] = scale * sum_i w[o][i] x[col][i]: an F32 weight [n_out][n_in] (the indexer's proj).
void dsa_gemv_f32(const float* w, const float* x, float* y, int64_t n_in, int64_t n_out, int64_t ncols, float scale,
                  void* stream);
/// The per-head Q8_0 products (k_b absorption, v_b): y[col][g][r] = W[g][r] . x[col][g], W Q8_0 GGUF blocks
/// [n_groups][rows][n_in], x the Q8_1 blocks of [ncols][n_groups][n_in] (native_quantize_q8_1), y f32;
/// n_in 256, 512 or 1024, ncols 1..8.
void dsa_grouped_q8_0(const void* w, const void* x_q8_1, float* y, int64_t n_in, int64_t rows, int64_t n_groups,
                      int64_t ncols, void* stream);

/// Scratch floats of dsa_attend for T tokens of up to `cap` selected cells.
uint64_t dsa_attend_scratch_floats(int64_t cap, int64_t T, const DsaShapes& s);
/// Softmax attention of T tokens over their selections: o[t][h] = sum_j p_j latent[ids[t][j]], p = softmax_j(
/// q_abs[t][h] . latent[ids[t][j]] * scale). q_abs [T][n_head][kv_lora] f32, o [T][n_head][kv_lora] f32. A cell
/// whose page is not resident (page_table -1) is masked. Capturable (grid sized by `cap`).
void dsa_attend(const DsaKvCache& c, const DsaShapes& s, const float* q_abs, const int32_t* ids, const int32_t* steps,
                int64_t T, int64_t cap, float scale, float* scratch, float* o, void* stream);

// ================================ the composed step ================================

/// The layer's weights, device pointers. Matrices are GGUF blocks of `*_type` (ggml type ids, any type
/// native_mmvq supports; GLM-5.3-Flash UD-IQ4_XS: all Q8_0 = 8), wk_b and wv_b Q8_0 only, idx_proj F32.
struct DsaWeights {
    const void* wq_a = nullptr;      int wq_a_type = 8;      ///< [n_embd -> q_lora]
    const float* q_a_norm = nullptr;                          ///< [q_lora]
    const void* wq_b = nullptr;      int wq_b_type = 8;      ///< [q_lora -> n_head * head_k]
    const void* wkv_a = nullptr;     int wkv_a_type = 8;     ///< [n_embd -> kv_lora] (no rope part)
    const float* kv_a_norm = nullptr;                         ///< [kv_lora]
    const void* wk_b = nullptr;                               ///< Q8_0 [n_head][kv_lora][head_k]
    const void* wv_b = nullptr;                               ///< Q8_0 [n_head][head_v][kv_lora]
    const void* wo = nullptr;        int wo_type = 8;        ///< [n_head * head_v -> n_embd]
    const void* idx_q_b = nullptr;   int idx_q_b_type = 8;   ///< [q_lora -> idx_n_head * idx_dim]
    const void* idx_k = nullptr;     int idx_k_type = 8;     ///< [n_embd -> idx_dim]
    const float* idx_k_norm = nullptr;                        ///< [idx_dim]
    const float* idx_k_norm_b = nullptr;                      ///< [idx_dim]
    const void* idx_gate = nullptr;  int idx_gate_type = 8;  ///< indexer_compressor_gate [n_embd -> idx_dim]
    const float* idx_ape = nullptr;                           ///< indexer_compressor_ape [kpool][idx_dim]
    const float* idx_proj = nullptr;                          ///< F32 [n_embd -> idx_n_head]
};

/// The step's intermediates, all device memory; after dsa_decode they hold the step's values (tests read them).
struct DsaScratch {
    void* xq = nullptr;       ///< Q8_1 of the widest activation (n_head * head_v x T, or n_head * kv_lora x T)
    float* qa = nullptr;      ///< [T][q_lora] before the norm
    float* qr = nullptr;      ///< [T][q_lora]
    float* q = nullptr;       ///< [T][n_head][head_k]
    float* q_abs = nullptr;   ///< [T][n_head][kv_lora]
    float* kv = nullptr;      ///< [T][kv_lora] before the norm
    float* latent = nullptr;  ///< [T][kv_lora]
    float* ik_raw = nullptr;  ///< [T][idx_dim] before the norm
    float* ik = nullptr;      ///< [T][idx_dim]
    float* ig = nullptr;      ///< [T][idx_dim]
    float* iw = nullptr;      ///< [T][idx_n_head] (scaled)
    float* iq = nullptr;      ///< [T][idx_n_head][idx_dim]
    float* scores = nullptr;  ///< [T][max_pools]
    int32_t* ids = nullptr;   ///< [T][cap]
    float* attn = nullptr;    ///< dsa_attend scratch
    float* o_lat = nullptr;   ///< [T][n_head][kv_lora]
    float* o = nullptr;       ///< [T][n_head][head_v]
    int64_t cap = 0, max_pools = 0, max_t = 0;
};
uint64_t dsa_scratch_bytes(const DsaShapes& s, int64_t max_t, int64_t max_pools);
void dsa_scratch_carve(void* base, const DsaShapes& s, int64_t max_t, int64_t max_pools, DsaScratch& sc);

/// One DSA layer for T <= 8 tokens: x [T][n_embd] f32 -> out [T][n_embd] f32, the cache appended and pooled, the
/// selection made (and resolved when streaming). `steps` [T][kStepCount] device, from dsa_step_fill.
/// `ids_override` (tests): when non-null, attend over these ids [T][cap] instead of the selection's.
void dsa_decode(const DsaWeights& wt, const DsaShapes& s, const DsaKvCache& c, const float* x, const int32_t* steps,
                int64_t T, const DsaScratch& sc, float* out, void* stream, const int32_t* ids_override = nullptr);

}  // namespace strata::kernels
