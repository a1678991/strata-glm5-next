// include/strata/core/glm_experts.hpp - GLM-5.3-Flash's MoE block for the decode window: the sigmoid router with the
// exp_probs_b selection bias (router_sigmoid.hpp), the clamped shared expert (ffn_clamp.hpp), and the routed experts
// from Strata's expert tiers - the VRAM cache (ExpertCache, filled in profile order), the page-locked RAM budget
// (FileExpertSource::pin_cache_complement, --resident-budget-gib) and the file tier (the GGUF shards in place) -
// computed on the GPU by native_expert_grouped with the clamped SwiGLU epilogue (NativeExpertLayout::swiglu_limit,
// swiglu_clamp_exp) and Q6_K down rows where the file has them.
//
// THE TIERS: an expert in the VRAM cache is computed there; an expert in the page-locked RAM tier is computed by the CPU
// pool when one is attached (set_pool; GlmPool, glm_pool.hpp), its rows uploaded once and summed by the combine; an
// expert from the file tier (and, without a pool, a RAM-tier one) is copied into a VRAM staging slot and computed on the
// GPU.  The order of one layer call: the router's ids, weights (and, with a pool, the activation) are read back while
// the GPU runs the shared expert; the plan; the file-tier staging; the GPU's grouped experts (VRAM + staged) launched;
// the CPU pool on the RAM-tier experts meanwhile; the CPU rows uploaded; the combine.  One wait for the read-back and
// one at the end of the layer; nothing here is captured in a graph.
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
class GlmPool;

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
        int64_t cpu = 0;              ///< ... of the RAM ones, computed by the CPU pool (not staged)
        double ms_pool = 0;           ///< host time in the CPU pool (activation quantization included)
        double ms_wait = 0;           ///< host time waiting for the GPU at the end of a layer call
        int64_t gpu_ready = 0;        ///< layer calls with CPU work whose GPU experts had finished when the pool did
        int64_t gpu_late = 0;         ///< ... had not (the GPU, not the pool, decided the layer's time)
    };

    GlmExperts() = default;
    ~GlmExperts();
    GlmExperts(const GlmExperts&) = delete;
    GlmExperts& operator=(const GlmExperts&) = delete;

    /// RAM-tier experts on the CPU pool (glm_pool.hpp); call before init.  Null (the default): every miss is staged.
    void set_pool(GlmPool* pool) { gp_ = pool; }
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
    GlmPool* gp_ = nullptr;
    void* cpu_dev_ = nullptr;            ///< the CPU rows on the device: map | rows (GlmPool::device_bytes)
    void* ev_ids_ = nullptr;             ///< cudaEvent_t: the router's read-back has landed
    void* ev_gpu_ = nullptr;             ///< cudaEvent_t: the GPU's grouped experts are done
    std::vector<int32_t> ent_;           ///< one expert's entries (host scratch)
    Stats st_;
};

}  // namespace strata::core
