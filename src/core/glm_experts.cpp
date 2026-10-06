// src/core/glm_experts.cpp - see include/strata/core/glm_experts.hpp.
#include "strata/core/glm_experts.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/glm_pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ffn_clamp.hpp"
#include "strata/kernels/glm_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace strata::core {
namespace sk = strata::kernels;

namespace {
constexpr uint64_t kAlign = 256;
uint64_t up(uint64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
}  // namespace

GlmExperts::~GlmExperts() {
    if (h_plan_) cudaFreeHost(h_plan_);
    if (h_ids_) cudaFreeHost(h_ids_);
    if (h_w_) cudaFreeHost(h_w_);
    if (ev_ids_) cudaEventDestroy((cudaEvent_t) ev_ids_);
    if (ev_gpu_) cudaEventDestroy((cudaEvent_t) ev_gpu_);
    if (arena_) cudaFree(arena_);
}

bool GlmExperts::init(const ModelGeometry& g, ExpertCache* cache, FileExpertSource* src, int max_t, std::string& err) {
    if (!g.glm()) { err = "glm experts: not a glm5-next geometry"; return false; }
    if (src == nullptr) { err = "glm experts: no expert source"; return false; }
    const auto& lay = sk::cpu::expert_layout();
    if (!lay.native || lay.n_layers != g.n_layers) {
        err = "glm experts: the native expert layout of the pack is not loaded";
        return false;
    }
    g_ = g;
    cache_ = cache;
    src_ = src;
    max_t_ = max_t;
    k_ = (int) g.n_expert_used;
    rs_.n_embd = (int) g.n_embd;
    rs_.n_expert = (int) g.n_expert;
    rs_.k = k_;
    rs_.norm = g.expert_weights_norm;
    rs_.scale = g.expert_weights_scale;
    if (g.gating != ExpertGating::Sigmoid) { err = "glm experts: only sigmoid gating is implemented"; return false; }
    lay_.assign((size_t) g.n_layers, sk::NativeExpertLayout{});
    uint64_t max_blob = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!lay.has_experts(l)) continue;
        const auto& f = lay.fmt[(size_t) l];
        if (!sk::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
            err = "glm experts: layer " + std::to_string(l) + ": no GPU expert kernel for types " +
                  std::to_string(f.gu_type) + "/" + std::to_string(f.d_type);
            return false;
        }
        sk::NativeExpertLayout L = sk::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        L.swiglu_limit = g.swiglu_clamp_exp;   // GLM: 10 (ggml_swiglu_clamp in every routed expert)
        lay_[(size_t) l] = L;
        max_blob = std::max<uint64_t>(max_blob, L.bytes);
    }
    const int64_t cap = (int64_t) max_t * k_, E = g.n_embd;
    slot_bytes_ = up(max_blob);
    const int64_t ff_sh = g.n_ff * std::max<int64_t>(1, g.n_expert_shared);
    uint64_t off = 0;
    auto take = [&](uint64_t bytes) { const uint64_t o = off; off = up(off + bytes); return o; };
    const uint64_t o_logits = take((uint64_t) max_t * g.n_expert * 4), o_w = take((uint64_t) cap * 4),
                   o_ids = take((uint64_t) cap * 4), o_shared = take((uint64_t) max_t * E * 4),
                   o_parts = take((uint64_t) cap * E * 4), o_xq = take((uint64_t) max_t * (E / 32) * 36),
                   o_scr = take(sk::native_expert_scratch_bytes(cap, g.n_ff)),
                   o_ffn = take(sk::ffn_clamp_scratch_bytes((int) E, (int) ff_sh, max_t)),
                   o_stage = take(slot_bytes_ * (uint64_t) cap);
    plan_bytes_ = up((uint64_t) cap * 8 + (uint64_t) (cap + 1) * 4 + 4 + (uint64_t) cap * 8);
    const uint64_t o_plan = take(plan_bytes_);
    const uint64_t o_cpu = gp_ != nullptr ? take(GlmPool::device_bytes(max_t, k_, E)) : 0;
    if (cudaMalloc(&arena_, off) != cudaSuccess) {
        cudaGetLastError();
        err = "glm experts: cudaMalloc of " + std::to_string(off >> 20) + " MiB (staging for " + std::to_string(cap) +
              " experts) failed";
        return false;
    }
    uint8_t* b = (uint8_t*) arena_;
    logits_ = (float*) (b + o_logits);
    w_ = (float*) (b + o_w);
    ids_ = (int32_t*) (b + o_ids);
    shared_ = (float*) (b + o_shared);
    parts_ = (float*) (b + o_parts);
    xq_ = b + o_xq;
    scratch_ = b + o_scr;
    ffn_scratch_ = b + o_ffn;
    staging_ = b + o_stage;
    plan_ = b + o_plan;
    cpu_dev_ = gp_ != nullptr ? b + o_cpu : nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (cudaEventCreateWithFlags(&e0, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&e1, cudaEventDisableTiming) != cudaSuccess) {
        err = "glm experts: cudaEventCreate";
        return false;
    }
    ev_ids_ = e0;
    ev_gpu_ = e1;
    ent_.reserve((size_t) cap);
    if (cudaHostAlloc(&h_plan_, plan_bytes_, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &h_ids_, (size_t) cap * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &h_w_, (size_t) cap * 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "glm experts: pinned staging";
        return false;
    }
    return true;
}

bool GlmExperts::moe(const GlmLayerWeights& w, const float* x, int T, float* out, void* stream, std::string& err) {
    cudaStream_t st = (cudaStream_t) stream;
    const int64_t l = w.layer, E = g_.n_embd, cap = (int64_t) max_t_ * k_;
    if (T < 1 || T > max_t_ || l < 0 || l >= g_.n_layers || lay_[(size_t) l].bytes == 0) {
        err = "glm experts: layer " + std::to_string(l) + " has no routed experts";
        return false;
    }
    const bool cpu = gp_ != nullptr && gp_->ready();
    // the router (F32 logits, sigmoid + bias selection, normalized and scaled weights); its ids and weights (and the
    // activation, for the CPU pool) go to the host while the GPU runs the shared expert
    sk::router_f32_logits(w.router, x, logits_, rs_, T, st);
    sk::router_sigmoid_topk(logits_, w.exp_probs_b, ids_, w_, rs_, T, st);
    cudaMemcpyAsync(h_ids_, ids_, (size_t) T * k_ * 4, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(h_w_, w_, (size_t) T * k_ * 4, cudaMemcpyDeviceToHost, st);
    if (cpu) cudaMemcpyAsync(gp_->host_x(), x, (size_t) T * E * 4, cudaMemcpyDeviceToHost, st);
    cudaEventRecord((cudaEvent_t) ev_ids_, st);
    sk::ffn_clamp(w.ffn, x, shared_, (int) E, (int) w.ffn_n_ff, T, g_.swiglu_clamp_shexp, ffn_scratch_, st);
    sk::quantize_q8_1_rows(x, T, E, xq_, st);
    if (const cudaError_t e = cudaEventSynchronize((cudaEvent_t) ev_ids_); e != cudaSuccess) {
        err = std::string("glm experts: router: ") + cudaGetErrorString(e);
        return false;
    }
    // the plan: one group per distinct expert the GPU computes (first-seen order), its entries (token t, rank j) ->
    // row t * k + j; the RAM-tier experts go to the CPU pool instead when it is attached
    const auto t0 = std::chrono::steady_clock::now();
    uint8_t* hp = (uint8_t*) h_plan_;
    auto* grp_ptr = (unsigned long long*) hp;
    auto* grp_start = (int32_t*) (hp + (size_t) cap * 8);
    auto* n_groups = grp_start + cap + 1;
    auto* ent_dst = n_groups + 1;
    auto* ent_tok = ent_dst + cap;
    std::vector<int32_t> experts;
    experts.reserve((size_t) cap);
    for (int i = 0; i < T * k_; ++i) {
        const int32_t e = h_ids_[i];
        if (e < 0 || e >= g_.n_expert) { err = "glm experts: router id out of range"; return false; }
        if (std::find(experts.begin(), experts.end(), e) == experts.end()) experts.push_back(e);
    }
    if (cpu) gp_->begin(l, T);
    int32_t n_ent = 0, n_staged = 0, ng = 0;
    const uint64_t blob = lay_[(size_t) l].bytes;
    for (size_t gi = 0; gi < experts.size(); ++gi) {
        const int32_t e = experts[gi];
        ent_.clear();
        for (int i = 0; i < T * k_; ++i)
            if (h_ids_[i] == e) ent_.push_back(i);
        const int32_t slot = cache_ != nullptr ? cache_->slot_of(l, e) : kNotResident;
        const bool ram = slot == kNotResident && src_->has_resident(l, e);
        if (cpu && ram) {   // the CPU pool: the RAM tier's pointer is stable (not the file tier's staging ring)
            const uint8_t* hb = src_->blob(l, e);
            if (hb == nullptr) {
                err = "glm experts: no blob for layer " + std::to_string(l) + " expert " + std::to_string(e);
                return false;
            }
            gp_->add(hb, ent_.data(), (int) ent_.size());
            ++st_.ram;
            ++st_.cpu;
            continue;
        }
        grp_start[ng] = n_ent;
        for (const int32_t i : ent_) {
            ent_dst[n_ent] = i;
            ent_tok[n_ent] = i / k_;
            ++n_ent;
        }
        if (slot != kNotResident) {
            grp_ptr[ng++] = (unsigned long long) (uintptr_t) cache_->device_slot(slot);
            ++st_.vram;
            continue;
        }
        const uint8_t* hb = src_->blob(l, e);
        if (hb == nullptr) {
            err = "glm experts: no blob for layer " + std::to_string(l) + " expert " + std::to_string(e);
            return false;
        }
        uint8_t* dst = staging_ + (uint64_t) n_staged * slot_bytes_;
        // synchronous: a file-tier blob lives in the source's staging ring, which a later blob() may reuse
        if (const cudaError_t ce = cudaMemcpy(dst, hb, blob, cudaMemcpyHostToDevice); ce != cudaSuccess) {
            err = std::string("glm experts: staging copy: ") + cudaGetErrorString(ce);
            return false;
        }
        grp_ptr[ng++] = (unsigned long long) (uintptr_t) dst;
        ++n_staged;
        st_.staged_bytes += blob;
        ++(ram ? st_.ram : st_.file);
    }
    grp_start[ng] = n_ent;
    *n_groups = ng;
    st_.ms_fetch += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    st_.groups += (int64_t) experts.size();
    st_.entries += T * k_;
    ++st_.windows;
    // the GPU's experts first, so that they run while the CPU pool works
    if (ng > 0) {
        cudaMemcpyAsync(plan_, h_plan_, plan_bytes_, cudaMemcpyHostToDevice, st);
        const uint8_t* dp = (const uint8_t*) plan_;
        const auto* d_ptr = (const unsigned long long*) dp;
        const auto* d_start = (const int32_t*) (dp + (size_t) cap * 8);
        const int32_t* d_n = d_start + cap + 1;
        const int32_t* d_dst = d_n + 1;
        const int32_t* d_tok = d_dst + cap;
        sk::native_expert_grouped(lay_[(size_t) l], d_ptr, d_start, d_n, d_dst, d_tok, cap, cap, xq_, scratch_, parts_,
                                  st);
    }
    const bool cpu_work = cpu && gp_->jobs() > 0;
    if (cpu_work) {
        cudaEventRecord((cudaEvent_t) ev_gpu_, st);
        cudaStreamQuery(st);   // submit what is queued before the host disappears into the pool
        const auto p0 = std::chrono::steady_clock::now();
        gp_->run();
        st_.ms_pool += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
        ++(cudaEventQuery((cudaEvent_t) ev_gpu_) == cudaSuccess ? st_.gpu_ready : st_.gpu_late);
        cudaMemcpyAsync(cpu_dev_, gp_->upload_src(), gp_->upload_bytes(), cudaMemcpyHostToDevice, st);
        const uint8_t* cd = (const uint8_t*) cpu_dev_;
        sk::glm_moe_combine(parts_, w_, shared_, out, (int) E, k_, T, st, (const float*) (cd + gp_->map_bytes()),
                            (const int32_t*) cd);
    } else {
        sk::glm_moe_combine(parts_, w_, shared_, out, (int) E, k_, T, st);
    }
    // h_plan_ and the pool's pinned rows are rewritten by the next call: their uploads must have landed
    const auto w0 = std::chrono::steady_clock::now();
    if (const cudaError_t e = cudaStreamSynchronize(st); e != cudaSuccess) {
        err = std::string("glm experts: ") + cudaGetErrorString(e);
        return false;
    }
    st_.ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    return true;
}

std::string GlmExperts::stats_line() const {
    char b[512];
    std::snprintf(b, sizeof b,
                  "glm experts: %lld layer calls, %lld routed entries, %lld experts computed: %lld from VRAM, %lld "
                  "from RAM, %lld from the file tier (%.1f MB staged over PCIe, %.1f ms fetching)",
                  (long long) st_.windows, (long long) st_.entries, (long long) st_.groups, (long long) st_.vram,
                  (long long) st_.ram, (long long) st_.file, (double) st_.staged_bytes / 1e6, st_.ms_fetch);
    std::string r = b;
    if (gp_ != nullptr && gp_->ready()) {
        const GlmPool::Stats& p = gp_->stats();
        std::snprintf(b, sizeof b,
                      "; CPU pool (%d threads): %lld of the RAM experts, %lld entries, %.1f MB read, %.1f ms (%.1f "
                      "quantizing), GPU done first in %lld of %lld layer calls; %.1f ms waiting for the GPU",
                      gp_->workers(), (long long) st_.cpu, (long long) p.entries, (double) p.bytes / 1e6,
                      st_.ms_pool, p.ms_quant, (long long) st_.gpu_ready, (long long) (st_.gpu_ready + st_.gpu_late),
                      st_.ms_wait);
        r += b;
    }
    return r;
}

}  // namespace strata::core
