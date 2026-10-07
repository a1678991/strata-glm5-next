// src/core/glm_pool.cpp - see include/strata/core/glm_pool.hpp.
#include "strata/core/glm_pool.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace strata::core {
namespace kc = strata::kernels::cpu;

namespace {
constexpr size_t kAlign = 256;
size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
}  // namespace

GlmPool::~GlmPool() {
    if (h_buf_) cudaFreeHost(h_buf_);
    if (h_x_) cudaFreeHost(h_x_);
}

size_t GlmPool::device_bytes(int max_t, int k, int64_t n_embd) {
    const size_t cap = (size_t) max_t * (size_t) k;
    return up(cap * sizeof(int32_t)) + cap * (size_t) n_embd * sizeof(float);
}

int GlmPool::workers() const { return pool_ ? pool_->workers() + (pool_->host_works() ? 1 : 0) : 0; }

bool GlmPool::init(const ModelGeometry& g, kc::ExpertPool* pool, int max_t, int k, std::string& err) {
    if (pool == nullptr) { err = "glm pool: no CPU expert pool"; return false; }
    if (!g.glm()) { err = "glm pool: not a glm5-next geometry"; return false; }
    if (max_t < 1 || max_t > kc::MAXT) { err = "glm pool: windows of 1.." + std::to_string(kc::MAXT) + " tokens"; return false; }
    if ((int64_t) max_t * k > (int64_t) kc::ExpertPool::kMaxSplitMulti) {
        err = "glm pool: more entries per window than the pool's split buffers";
        return false;
    }
    if (!kc::native_experts_available()) { err = "glm pool: this build has no ggml-cpu expert path"; return false; }
    const auto& lay = kc::expert_layout();
    if (!lay.native || lay.n_layers != g.n_layers) { err = "glm pool: the native expert layout is not loaded"; return false; }
    fmt_.assign((size_t) g.n_layers, kc::NativeFmt{});
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!lay.has_experts(l)) continue;
        const kc::NativeFmt& f = lay.fmt[(size_t) l];
        kc::NativeFmt chk;
        if (!kc::native_fmt(f.gu_type, f.d_type, f.n_embd, f.n_ff, chk, err)) {
            err = "glm pool: layer " + std::to_string(l) + ": " + err;
            return false;
        }
        if (chk.bytes != f.bytes || f.n_embd != g.n_embd || f.n_ff > kc::kNativeMaxFF ||
            kc::q2_native_kernels(f.gu_type) || kc::q2_native_kernels(f.d_type)) {
            err = "glm pool: layer " + std::to_string(l) + ": an expert format the pool does not take for glm5-next";
            return false;
        }
        fmt_[(size_t) l] = chk;
    }
    pool_ = pool;
    limit_ = g.swiglu_clamp_exp;
    max_t_ = max_t;
    k_ = k;
    E_ = g.n_embd;
    const size_t cap = (size_t) max_t * (size_t) k;
    map_bytes_ = up(cap * sizeof(int32_t));
    if (cudaHostAlloc(&h_buf_, device_bytes(max_t, k, E_), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &h_x_, (size_t) max_t * (size_t) E_ * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        err = "glm pool: pinned host rows";
        return false;
    }
    map_ = (int32_t*) h_buf_;
    rows_ = (float*) ((uint8_t*) h_buf_ + map_bytes_);
    nact_.assign((size_t) max_t * kc::kNativeActBytes, 0);
    tok_used_.assign((size_t) max_t, 0);
    jobs_.assign(cap, kc::ExpertJobMulti{});
    return true;
}

void GlmPool::begin(int64_t layer, int T) {
    layer_ = layer;
    T_ = T;
    njobs_ = 0;
    nrows_ = 0;
    for (int i = 0; i < T * k_; ++i) map_[i] = -1;
    std::fill(tok_used_.begin(), tok_used_.end(), (uint8_t) 0);
}

void GlmPool::add(const uint8_t* blob, const int32_t* entries, int n) {
    kc::ExpertJobMulti& j = jobs_[(size_t) njobs_++];
    j = kc::ExpertJobMulti{};
    j.blob = blob;
    for (int q = 0; q < n; ++q) {
        const int i = entries[q], t = i / k_;
        map_[i] = nrows_;
        j.nact[j.nt] = nact_.data() + (size_t) t * kc::kNativeActBytes;
        j.out[j.nt] = rows_ + (size_t) nrows_ * (size_t) E_;
        ++j.nt;
        ++nrows_;
        tok_used_[(size_t) t] = 1;
    }
}

void GlmPool::run() {
    if (njobs_ == 0) return;
    const kc::NativeFmt& f = fmt_[(size_t) layer_];
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < T_; ++t)
        if (tok_used_[(size_t) t])
            kc::native_quant_act(f, h_x_ + (size_t) t * (size_t) E_, nact_.data() + (size_t) t * kc::kNativeActBytes);
    const auto t1 = std::chrono::steady_clock::now();
    pool_->run_split_multi_native(f, jobs_.data(), njobs_, limit_);
    const auto t2 = std::chrono::steady_clock::now();
    st_.ms_quant += std::chrono::duration<double, std::milli>(t1 - t0).count();
    st_.ms_run += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ++st_.calls;
    st_.experts += njobs_;
    st_.entries += nrows_;
    st_.bytes += (uint64_t) njobs_ * (uint64_t) f.bytes;
}

}  // namespace strata::core
