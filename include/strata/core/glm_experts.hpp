// include/strata/core/glm_experts.hpp - GLM-5.3-Flash's MoE block for the decode window: the sigmoid router with the
// exp_probs_b selection bias (router_sigmoid.hpp), the clamped shared expert (ffn_clamp.hpp), and the routed experts
// from Strata's expert tiers - the VRAM cache (ExpertCache, filled in profile order), the page-locked RAM budget
// (FileExpertSource::pin_cache_complement, --resident-budget-gib) and the file tier (the GGUF shards in place) -
// computed on the GPU by native_expert_grouped with the clamped SwiGLU epilogue (NativeExpertLayout::swiglu_limit,
// swiglu_clamp_exp) and Q6_K down rows where the file has them.
//
// WHAT DIFFERS FROM THE QWEN PATH, deliberately for now: an expert the VRAM cache does not hold is copied from
// its tier (RAM or file) into a VRAM staging slot and computed there - the verify window's "PCIe share" applied to
// every miss - instead of going to the CPU pool.  The CPU pool's clamped rows exist (native_gu_rows_clamp)
// and are not wired for GLM yet.  The router's ids are read back to the host once per MoE layer (a synchronization
// per layer); nothing here is captured in a graph.
//
// Order of the result, as llama.cpp f1cee99 (build_moe_ffn; G:640-657): out = sum_j w_j expert_j(x) + shexp(x)
// (glm_moe.hpp).
#pragma once

#include "strata/core/glm.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/router_sigmoid.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class ExpertCache;
class FileExpertSource;

class GlmExperts {
public:
    struct Stats {
        int64_t windows = 0;          ///< MoE layer calls
        int64_t entries = 0;          ///< routed (token, expert) entries
        int64_t groups = 0;           ///< distinct experts computed (per layer call)
        int64_t vram = 0;             ///< ... of them held by the VRAM cache
        int64_t ram = 0;              ///< ... copied from the page-locked RAM tier
        int64_t file = 0;             ///< ... read from the file tier
        uint64_t staged_bytes = 0;    ///< bytes copied into VRAM staging (RAM + file)
        double ms_fetch = 0;          ///< host time spent fetching and staging the misses
    };

    GlmExperts() = default;
    ~GlmExperts();
    GlmExperts(const GlmExperts&) = delete;
    GlmExperts& operator=(const GlmExperts&) = delete;

    /// `cache` may be null (no VRAM tier); `src` holds every expert (RAM budget and file tier).  The per-layer formats
    /// come from the loaded expert layout (kernels::cpu::expert_layout(), native).
    bool init(const ModelGeometry& g, ExpertCache* cache, FileExpertSource* src, int max_t, std::string& err);
    /// out (T x n_embd) = the MoE block of layer w.layer on x (T x n_embd, the ffn_norm output).  Synchronizes once
    /// (the router's ids).
    bool moe(const GlmLayerWeights& w, const float* x, int T, float* out, void* stream, std::string& err);

    const Stats& stats() const { return st_; }
    /// The tiers line: "glm experts: N entries, G experts: V vram, R ram, F file (P MB staged, X ms)".
    std::string stats_line() const;
    /// The last call's ids and weights (host, [T][k]), for routing checks against llama.cpp's ffn_moe_topk.
    const int32_t* last_ids() const { return h_ids_; }
    const float* last_weights() const { return h_w_; }

private:
    ModelGeometry g_;
    ExpertCache* cache_ = nullptr;
    FileExpertSource* src_ = nullptr;
    int max_t_ = 0, k_ = 0;
    strata::kernels::SigmoidRouterShape rs_;
    std::vector<strata::kernels::NativeExpertLayout> lay_;   ///< per layer (MoE layers)
    void* arena_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr;
    int32_t* ids_ = nullptr;
    void *xq_ = nullptr, *scratch_ = nullptr, *ffn_scratch_ = nullptr;
    uint8_t* staging_ = nullptr;          ///< max_t * k slots of the largest blob
    uint64_t slot_bytes_ = 0;
    // the plan: grp_ptr [cap] u64 | grp_start [cap + 1] | n_groups [1] | ent_dst [cap] | ent_tok [cap], device and pinned
    void* plan_ = nullptr;
    void* h_plan_ = nullptr;
    uint64_t plan_bytes_ = 0;
    int32_t* h_ids_ = nullptr;
    float* h_w_ = nullptr;
    Stats st_;
};

}  // namespace strata::core
