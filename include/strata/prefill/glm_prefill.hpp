// include/strata/prefill/glm_prefill.hpp - GLM-5.3-Flash (llama.cpp arch `glm5-next`): reading a prompt in chunks.
// The GLM path beside the Qwen prompt path
// (include/strata/prefill/prefill.hpp), which it does not touch.
//
// THE REFERENCE is llama.cpp f1cee99 reading a whole batch: src/models/glm5-next.cpp ("G") with n_tokens > 1.
// A chunk of T tokens at positions p0 .. p0 + T - 1 goes through every layer at once, writing the session of a
// GlmEngine in place (GlmSessionView): after the last chunk the decode windows continue as if they had read the
// prompt themselves.  Per layer:
//
//   mHC            mhc_flat_norm, the hc_*_fn product as a GEMM, mhc_pre, the norm, mhc_post: the decode kernels,
//                  which take any number of tokens
//   KDA            the q|k|v, f_a/f_b, g_a/g_b, beta and W_o products as GEMMs; the conv and the recurrence walk the
//                  chunk token by token inside one launch each (kda_conv_l2_prompt / kda_step_norm_prompt: the decode
//                  step's per-token arithmetic), the history and the state carried from chunk to chunk.  llama.cpp
//                  reads a prompt with the same sequential recurrence (its fused gated_delta_net, CPU and CUDA; the
//                  chunked build_delta_net_chunking is not used), so the numerics match the reference's form
//   DSA            the projections as GEMMs; the chunk's latents and key | gate rows appended to the cache, the pools
//                  the chunk completes pooled (G:801-829), then for every query of the chunk the lightning-indexer
//                  scores over the pools that end at or before it and llama.cpp's selection - every cell while
//                  pos + 1 <= 2,051, else the top 512 pools plus its tail (dsa_index_scores / dsa_select, the decode
//                  kernels) - and the attention over the selection (glm::dsa_prompt_attend, an online softmax over
//                  the cached latent); k_b and v_b per head as GEMMs.  The queries go through in sub-batches of 256
//   dense FFN      gate, up and down as GEMMs, the clamped SwiGLU (swiglu_clamp_shexp) between them
//   MoE            the sigmoid router over the chunk (router_f32_logits / router_sigmoid_topk: the decode kernels),
//                  the shared expert as GEMMs, then every expert the chunk routes to, in id order: its rows gathered,
//                  its gate|up and down dequantized to FP16 and multiplied (two GEMMs over its rows) with the clamped
//                  SwiGLU (swiglu_clamp_exp) between them, and its weighted rows added into the output.  An expert the
//                  VRAM cache holds is read from its slot; any other is copied from its tier (page-locked RAM by DMA,
//                  else through a page-locked staging buffer from the RAM copy or the GGUF in place) into a ring of
//                  VRAM slots on a copy stream, while the GPU computes the experts before it
//
// Every quantized projection is an FP16 tensor-core GEMM (GGUF blocks dequantized to FP16, activations rounded to
// FP16, F32 accumulation): Strata's Qwen prompt path does the same.  The decode path quantizes activations to Q8_1
// instead, and llama.cpp's CPU to Q8_0 / Q8_K, so the three differ by rounding, not by algorithm.
//
// Memory: buffers for chunks of `chunk` tokens (bytes_needed) plus a dequantization scratch of the largest weight
// group and the miss ring, allocated by `init`; nothing is borrowed from the expert cache yet.
#pragma once

#include "strata/core/glm.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {
class ExpertCache;
class FileExpertSource;
}  // namespace strata::core

namespace strata::prefill {

struct GlmPrefillStats {
    int64_t tokens = 0, chunks = 0;
    double ms_total = 0;
    int64_t moe_calls = 0;        ///< MoE layer passes (layers x chunks)
    int64_t entries = 0;          ///< routed (token, expert) entries
    int64_t experts = 0;          ///< distinct experts computed (summed over the MoE layer passes)
    int64_t vram = 0, ram = 0, file = 0;   ///< ... of them from the VRAM cache, the RAM tier, the file tier
    uint64_t staged_bytes = 0;    ///< bytes copied host -> device for the misses
    double ms_copy_host = 0;      ///< host time copying misses into the page-locked staging
    double ms_wait_slot = 0;      ///< host time waiting for a free ring slot (the GPU behind)
    double ms_route_sync = 0;     ///< host time waiting for the router's ids
};

class GlmPrefill {
public:
    GlmPrefill();
    ~GlmPrefill();
    GlmPrefill(const GlmPrefill&) = delete;
    GlmPrefill& operator=(const GlmPrefill&) = delete;

    /// Device bytes `init` allocates for chunks of `chunk` tokens on this geometry and session (a context of
    /// cfg.max_cells cells), the miss ring included.
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::GlmConfig& cfg, int64_t chunk);

    /// `eng` must be initialized; `cache` (may be null) and `src` (may be null while no MoE layer runs) are the tiers
    /// the decode path uses.  The prompt runs on the engine's stream.
    bool init(core::GlmEngine& eng, core::ExpertCache* cache, core::FileExpertSource* src, int64_t chunk,
              std::string& err);

    /// Tokens tokens[0 .. n) at positions pos0 .. pos0 + n - 1, in chunks.  The session must hold exactly the cells
    /// before pos0.  With the whole model, the logits of the last min(n_logits, max_t, n) positions are left in the
    /// engine (GlmEngine::head_rows).  Synchronizes before returning.
    bool run(const int32_t* tokens, int64_t n, int64_t pos0, int n_logits, std::string& err);

    /// Append x after `layer` ([T][hc][n_embd] per chunk, GLMT rows in position order) to `w` (null stops it).
    void set_dump(int64_t layer, core::GlmtWriter* w);

    int64_t chunk() const;
    uint64_t device_bytes() const;
    const GlmPrefillStats& stats() const;
    /// "glm prefill: N tokens in C chunks, X tok/s; experts: ..."
    std::string stats_line() const;
    /// STRATA_GLM_PREFILL_TIMING=1: GPU time per phase, one line; empty when off.
    std::string timing_report() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::prefill
