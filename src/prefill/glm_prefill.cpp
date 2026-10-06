// src/prefill/glm_prefill.cpp - GLM-5.3-Flash prompt path: see include/strata/prefill/glm_prefill.hpp.
#include "strata/prefill/glm_prefill.hpp"

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kda_decode.hpp"
#include "strata/kernels/mhc.hpp"
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/prefill/glm_prefill_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::prefill {
namespace {

namespace sk = strata::kernels;
namespace sc = strata::core;
using Clock = std::chrono::steady_clock;

constexpr uint64_t kAlign = 256;
constexpr int64_t kDsaSub = 256;    // DSA queries per sub-batch (q, q_abs, scores, ids, the attention output)
constexpr int64_t kFfnSub = 1024;   // dense-FFN rows per sub-batch (the F32 gate | up of 12,288 + 12,288)
constexpr int kRing = 6;            // VRAM slots (and page-locked host slots) for the experts the cache misses

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Carver {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <class T>
    T* take(uint64_t count) {
        T* p = base ? (T*) (base + used) : nullptr;
        used = (used + count * sizeof(T) + kAlign - 1) / kAlign * kAlign;
        return p;
    }
};

bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("glm prefill: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

/// Every dimension the buffers depend on.
struct Dims {
    int64_t T = 0, E = 0, HC = 0, MIX = 0;
    int64_t DI = 0, NH = 0, HD = 0;                    // KDA
    int64_t NFF_D = 0, NFF = 0, FF_SH = 0, NEXP = 0, K = 0;
    int64_t QL = 0, KVL = 0, DH = 0, HK = 0, HV = 0, IH = 0, ID = 0, KP = 0, max_pools = 0, cap = 0;   // DSA
    bool any_dsa = false, any_moe = false, any_dense = false;
    uint64_t max_blob = 0;
    int64_t w_elems = 0, dsaw_elems = 0;
};

Dims dims_of(const sc::ModelGeometry& g, const sc::GlmConfig& cfg, int64_t chunk) {
    Dims d;
    d.T = chunk;
    d.E = g.n_embd;
    d.HC = g.hc;
    d.MIX = (2 + g.hc) * g.hc;
    d.DI = g.ssm_qk_dim;
    d.NH = g.ssm_v_heads;
    d.HD = g.ssm_state_size;
    d.NFF_D = g.n_ff_dense;
    d.NFF = g.n_ff;
    d.FF_SH = g.n_ff * std::max<int64_t>(1, g.n_expert_shared);
    d.NEXP = g.n_expert;
    d.K = g.n_expert_used;
    sk::DsaShapes s;
    s.n_head = g.n_head;
    s.q_lora = g.q_lora;
    s.kv_lora = g.kv_lora;
    s.head_k = g.head_dim - g.n_rot;
    s.head_v = g.head_dim_v;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    s.kpool = g.idx_kpool;
    s.idx_top_k = g.idx_top_k;
    d.QL = s.q_lora;
    d.KVL = s.kv_lora;
    d.DH = s.n_head;
    d.HK = s.head_k;
    d.HV = s.head_v;
    d.IH = s.idx_n_head;
    d.ID = s.idx_dim;
    d.KP = s.kpool;
    const int64_t n_run = cfg.n_run < 0 || cfg.n_run > g.n_layers ? g.n_layers : cfg.n_run;
    for (int64_t l = 0; l < n_run; ++l) {
        if (g.attn_layer(l)) d.any_dsa = true;
        if (l < g.n_dense_lead) d.any_dense = true;
        else d.any_moe = true;
    }
    if (d.any_dsa) {
        d.max_pools = sk::dsa_max_pools(cfg.max_cells, s);
        d.cap = sk::dsa_sel_cap(s);
    }
    const auto& lay = sk::cpu::expert_layout();
    if (d.any_moe && lay.native) d.max_blob = lay.max_blob;
    // the dequantization scratch: the largest group of weights dequantized at once (KDA q|k|v, the dense FFN's gate
    // and up, an MoE expert's gate|up, a DSA layer's attn_output, the head-wise products' inputs)
    d.w_elems = std::max<int64_t>({3 * d.DI * d.E, d.any_dense ? 2 * d.NFF_D * d.E : 0, 2 * d.FF_SH * d.E,
                                   2 * d.NFF * d.E, d.any_dsa ? d.DH * d.HV * d.E : 0, d.HC * d.E * d.MIX});
    d.dsaw_elems = d.any_dsa ? d.DH * d.HK * d.QL + d.IH * d.ID * d.QL + d.DH * d.KVL * d.HK + d.DH * d.HV * d.KVL : 0;
    return d;
}

/// The device buffers of a chunk.  Persistent ones live for the whole chunk; the phase regions (mHC, KDA, dense FFN,
/// MoE, DSA) share one region, since a layer runs one phase at a time.
struct Bufs {
    // persistent
    int32_t *tok = nullptr, *steps = nullptr;
    float *x = nullptr, *cur = nullptr, *xn = nullptr, *f = nullptr, *mixes = nullptr, *post = nullptr, *comb = nullptr;
    uint16_t* xn16 = nullptr;
    // mHC
    float* flat = nullptr;
    uint16_t* flat16 = nullptr;
    // KDA
    float *qkv = nullptr, *fb = nullptr, *gb = nullptr, *braw = nullptr, *fa = nullptr, *ga = nullptr, *y = nullptr;
    uint16_t *fa16 = nullptr, *ga16 = nullptr, *y16 = nullptr;
    // dense FFN
    float* gu_d = nullptr;
    uint16_t* h_d = nullptr;
    // MoE
    float *logits = nullptr, *rw = nullptr, *gu = nullptr, *shared = nullptr, *GU = nullptr, *D = nullptr, *w_sorted = nullptr;
    int32_t *rids = nullptr, *tok_sorted = nullptr;
    uint16_t *sh16 = nullptr, *X16 = nullptr, *H16 = nullptr;
    // DSA
    float *qa = nullptr, *qr = nullptr, *kv = nullptr, *lat = nullptr, *ikr = nullptr, *ik = nullptr, *ig = nullptr,
          *iw = nullptr;
    uint16_t *qr16 = nullptr, *o16 = nullptr;
    float *q = nullptr, *q_abs = nullptr, *iq = nullptr, *scores = nullptr, *olat = nullptr, *ov = nullptr;
    uint16_t *q16 = nullptr, *olat16 = nullptr;
    int32_t* ids = nullptr;
    // weights
    uint16_t *wscr = nullptr, *dsaw = nullptr;
    // the miss ring
    uint8_t* ring = nullptr;
    uint64_t bytes = 0;
};

Bufs carve(const Dims& d, void* base) {
    Bufs b;
    Carver c{(uint8_t*) base, 0};
    const int64_t T = d.T;
    b.tok = c.take<int32_t>(T);
    b.steps = c.take<int32_t>((uint64_t) T * sk::kStepCount);
    b.x = c.take<float>((uint64_t) T * d.HC * d.E);
    b.cur = c.take<float>((uint64_t) T * d.E);
    b.xn = c.take<float>((uint64_t) T * d.E);
    b.f = c.take<float>((uint64_t) T * d.E);
    b.xn16 = c.take<uint16_t>((uint64_t) T * d.E);
    b.mixes = c.take<float>((uint64_t) T * d.MIX);
    b.post = c.take<float>((uint64_t) T * d.HC);
    b.comb = c.take<float>((uint64_t) T * d.HC * d.HC);
    const uint64_t phase0 = c.used;
    uint64_t phase_max = 0;
    auto phase = [&](auto&& fill) {
        Carver p{c.base, phase0};
        fill(p);
        phase_max = std::max(phase_max, p.used - phase0);
    };
    phase([&](Carver& p) {   // mHC
        b.flat = p.take<float>((uint64_t) T * d.HC * d.E);
        b.flat16 = p.take<uint16_t>((uint64_t) T * d.HC * d.E);
    });
    phase([&](Carver& p) {   // KDA (y16 reuses the conv output's region once the recurrence has read it)
        b.qkv = p.take<float>((uint64_t) T * 3 * d.DI);
        b.y16 = (uint16_t*) b.qkv;
        b.fb = p.take<float>((uint64_t) T * d.DI);
        b.gb = p.take<float>((uint64_t) T * d.DI);
        b.y = p.take<float>((uint64_t) T * d.DI);
        b.braw = p.take<float>((uint64_t) T * d.NH);
        b.fa = p.take<float>((uint64_t) T * d.HD);
        b.ga = p.take<float>((uint64_t) T * d.HD);
        b.fa16 = p.take<uint16_t>((uint64_t) T * d.HD);
        b.ga16 = p.take<uint16_t>((uint64_t) T * d.HD);
    });
    if (d.any_dense)
        phase([&](Carver& p) {
            b.gu_d = p.take<float>((uint64_t) std::min(T, kFfnSub) * 2 * d.NFF_D);
            b.h_d = p.take<uint16_t>((uint64_t) T * d.NFF_D);
        });
    if (d.any_moe)
        phase([&](Carver& p) {
            b.logits = p.take<float>((uint64_t) T * d.NEXP);
            b.rids = p.take<int32_t>((uint64_t) T * d.K);
            b.rw = p.take<float>((uint64_t) T * d.K);
            b.tok_sorted = p.take<int32_t>((uint64_t) T * d.K);
            b.w_sorted = p.take<float>((uint64_t) T * d.K);
            b.gu = p.take<float>((uint64_t) T * 2 * std::max(d.FF_SH, d.NFF));   // the shared expert's, then GU
            b.GU = b.gu;
            b.sh16 = p.take<uint16_t>((uint64_t) T * std::max(d.FF_SH, d.NFF));
            b.H16 = b.sh16;
            b.shared = p.take<float>((uint64_t) T * d.E);
            b.X16 = p.take<uint16_t>((uint64_t) T * d.E);
            b.D = p.take<float>((uint64_t) T * d.E);
        });
    if (d.any_dsa)
        phase([&](Carver& p) {
            const int64_t S = std::min(T, kDsaSub);
            b.qa = p.take<float>((uint64_t) T * d.QL);
            b.qr = p.take<float>((uint64_t) T * d.QL);
            b.qr16 = p.take<uint16_t>((uint64_t) T * d.QL);
            b.kv = p.take<float>((uint64_t) T * d.KVL);
            b.lat = p.take<float>((uint64_t) T * d.KVL);
            b.ikr = p.take<float>((uint64_t) T * d.ID);
            b.ik = p.take<float>((uint64_t) T * d.ID);
            b.ig = p.take<float>((uint64_t) T * d.ID);
            b.iw = p.take<float>((uint64_t) T * d.IH);
            b.o16 = p.take<uint16_t>((uint64_t) T * d.DH * d.HV);
            b.q = p.take<float>((uint64_t) S * d.DH * d.HK);
            b.q16 = p.take<uint16_t>((uint64_t) S * d.DH * d.HK);
            b.q_abs = p.take<float>((uint64_t) S * d.DH * d.KVL);
            b.iq = p.take<float>((uint64_t) S * d.IH * d.ID);
            b.scores = p.take<float>((uint64_t) S * d.max_pools);
            b.ids = p.take<int32_t>((uint64_t) S * d.cap);
            b.olat = p.take<float>((uint64_t) S * d.DH * d.KVL);
            b.olat16 = p.take<uint16_t>((uint64_t) S * d.DH * d.KVL);
            b.ov = p.take<float>((uint64_t) S * d.DH * d.HV);
        });
    c.used = phase0 + phase_max;
    b.wscr = c.take<uint16_t>((uint64_t) d.w_elems);
    if (d.dsaw_elems > 0) b.dsaw = c.take<uint16_t>((uint64_t) d.dsaw_elems);
    if (d.max_blob > 0) b.ring = c.take<uint8_t>((uint64_t) kRing * ((d.max_blob + kAlign - 1) / kAlign * kAlign));
    b.bytes = c.used + kAlign;
    return b;
}

// STRATA_GLM_PREFILL_TIMING=1: events on the stream; the time between consecutive marks is charged to the phase of
// the first (a GPU idle gap, e.g. the host planning the experts, lands on the phase that waits).  Folded whenever the
// host has synchronized the stream.
enum Phase { kEmbed, kMhc, kKdaProj, kKdaConv, kKdaRec, kKdaOut, kDsaProj, kDsaCache, kDsaQ, kDsaIdx, kDsaAttn,
             kDsaOut, kFfn, kRouter, kShared, kExpGemm, kExpWait, kHead, kPhaseCount };
const char* const kPhaseNames[kPhaseCount] = {"embed", "mhc", "kda proj", "kda conv", "kda recurrence", "kda out",
                                              "dsa proj", "dsa cache", "dsa q", "dsa select", "dsa attention",
                                              "dsa out", "dense ffn", "router", "shared expert", "experts",
                                              "expert wait", "head"};
struct Timer {
    bool on = false;
    std::vector<cudaEvent_t> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPhaseCount] = {};
    void mark(int phase, cudaStream_t s) {
        if (!on) return;
        if (used == ev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        cudaEventRecord(ev[used], s);
        ++used;
    }
    void fold() {   // every mark has completed
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, ev[i], ev[i + 1]) == cudaSuccess) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~Timer() {
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
};

}  // namespace

struct GlmPrefill::Impl {
    sc::GlmEngine* eng = nullptr;
    sc::GlmSessionView v;
    const sc::ModelGeometry* g = nullptr;
    sc::ExpertCache* cache = nullptr;
    sc::FileExpertSource* src = nullptr;
    Dims d;
    Bufs b;
    void* arena = nullptr;
    cudaStream_t st = nullptr, copy = nullptr;
    glm::Gemm16 gemm;
    sk::SigmoidRouterShape rs;
    // host
    int32_t *h_tok = nullptr, *h_steps = nullptr, *h_ids = nullptr, *h_tok_sorted = nullptr;
    float *h_w = nullptr, *h_w_sorted = nullptr;
    uint8_t* h_ring = nullptr;
    uint64_t ring_slot = 0;
    cudaEvent_t ev_route = nullptr;
    cudaEvent_t ring_copied[kRing] = {}, ring_used[kRing] = {};
    bool ring_live[kRing] = {};
    int64_t ring_next = 0;
    std::vector<sk::NativeExpertLayout> lay;
    std::vector<sc::GlmtWriter*> dump;
    std::vector<float> h_dump;
    GlmPrefillStats stats;
    Timer tm;

    ~Impl() {
        if (st) cudaStreamSynchronize(st);
        if (copy) {
            cudaStreamSynchronize(copy);
            cudaStreamDestroy(copy);
        }
        for (int k = 0; k < kRing; ++k) {
            if (ring_copied[k]) cudaEventDestroy(ring_copied[k]);
            if (ring_used[k]) cudaEventDestroy(ring_used[k]);
        }
        if (ev_route) cudaEventDestroy(ev_route);
        for (void* p : {(void*) h_tok, (void*) h_steps, (void*) h_ids, (void*) h_tok_sorted, (void*) h_w,
                        (void*) h_w_sorted, (void*) h_ring})
            if (p) cudaFreeHost(p);
        if (arena) cudaFree(arena);
    }

    // Y[T][N] (row stride ldy) = X16[T][K] (row stride ldx) . W^T, W the GGUF blocks of `type`: dequantized to FP16
    // in the scratch (in row slices when it does not fit), then the GEMM
    bool proj(const void* W, int type, const uint16_t* X, int64_t ldx, float* Y, int64_t ldy, int64_t T, int64_t N,
              int64_t K, std::string& err) {
        if (W == nullptr || !sk::dequant_bf16_supported(type)) {
            err = "glm prefill: a projection of ggml type " + std::to_string(type) + " has no FP16 dequantizer";
            return false;
        }
        const int64_t rows = std::max<int64_t>(1, std::min<int64_t>(N, d.w_elems / K));
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            sk::dequant_f16(type, W, r0, n, K, b.wscr, st);
            gemm.run(X, ldx, b.wscr, Y + r0, ldy, T, n, K);
        }
        return true;
    }

    void mhc_half(const sc::GlmLayerWeights& w, bool ffn, int64_t T, std::string& err, bool& ok) {
        const int64_t hcd = d.HC * d.E;
        tm.mark(kMhc, st);
        sk::mhc_flat_norm(b.x, b.flat, v.mshape, (int) T, st);
        glm::to_f16(b.flat, b.flat16, T * hcd, st);
        ok = proj(ffn ? w.hc_ffn_fn : w.hc_attn_fn, ffn ? w.hc_ffn_fn_type : w.hc_attn_fn_type, b.flat16, hcd, b.mixes,
                  d.MIX, T, d.MIX, hcd, err);
        if (!ok) return;
        sk::mhc_pre(b.x, b.mixes, ffn ? w.hc_ffn_scale : w.hc_attn_scale, ffn ? w.hc_ffn_base : w.hc_attn_base, b.cur,
                    b.post, b.comb, v.mshape, (int) T, st);
        sk::dsa_rms_norm(b.cur, ffn ? w.ffn_norm : w.attn_norm, g->norm_rms_eps, d.E, T, b.xn, st);
        glm::to_f16(b.xn, b.xn16, T * d.E, st);
    }

    bool kda(int64_t l, const sc::GlmLayerWeights& w, int64_t T, std::string& err) {
        const int64_t E = d.E, DI = d.DI, HD = d.HD, NH = d.NH;
        tm.mark(kKdaProj, st);
        // q | k | v: the three [d_inner][n_embd] matrices dequantized back to back = one [3 d_inner][n_embd] GEMM
        if (w.wq_type != w.wk_type || w.wq_type != w.wv_type || 3 * DI * E > d.w_elems) {
            if (!proj(w.wq, w.wq_type, b.xn16, E, b.qkv, 3 * DI, T, DI, E, err) ||
                !proj(w.wk, w.wk_type, b.xn16, E, b.qkv + DI, 3 * DI, T, DI, E, err) ||
                !proj(w.wv, w.wv_type, b.xn16, E, b.qkv + 2 * DI, 3 * DI, T, DI, E, err))
                return false;
        } else {
            sk::dequant_f16(w.wq_type, w.wq, 0, DI, E, b.wscr, st);
            sk::dequant_f16(w.wk_type, w.wk, 0, DI, E, b.wscr + DI * E, st);
            sk::dequant_f16(w.wv_type, w.wv, 0, DI, E, b.wscr + 2 * DI * E, st);
            gemm.run(b.xn16, E, b.wscr, b.qkv, 3 * DI, T, 3 * DI, E);
        }
        if (!proj(w.f_a, w.f_a_type, b.xn16, E, b.fa, HD, T, HD, E, err) ||
            !proj(w.g_a, w.g_a_type, b.xn16, E, b.ga, HD, T, HD, E, err) ||
            !proj(w.beta, w.beta_type, b.xn16, E, b.braw, NH, T, NH, E, err))
            return false;
        glm::to_f16(b.fa, b.fa16, T * HD, st);
        glm::to_f16(b.ga, b.ga16, T * HD, st);
        if (!proj(w.f_b, w.f_b_type, b.fa16, HD, b.fb, DI, T, DI, HD, err) ||
            !proj(w.g_b, w.g_b_type, b.ga16, HD, b.gb, DI, T, DI, HD, err))
            return false;
        // the conv (history carried from the previous chunk, in place: h over qkv) and the recurrence
        tm.mark(kKdaConv, st);
        sk::kda_conv_l2_prompt(v.kda_hist[(size_t) l], b.qkv, w.conv_q, w.conv_k, w.conv_v, b.qkv, v.kshape, T, st);
        tm.mark(kKdaRec, st);
        sk::kda_step_norm_prompt(v.kda_state[(size_t) l], b.qkv, b.fb, w.dt_b, w.ssm_a, b.braw, b.gb, w.ssm_norm, b.y,
                                 v.kshape, T, st);
        tm.mark(kKdaOut, st);
        glm::to_f16(b.y, b.y16, T * DI, st);
        return proj(w.wo, w.wo_type, b.y16, DI, b.f, E, T, E, DI, err);
    }

    bool dense_ffn(const sc::GlmLayerWeights& w, int64_t T, std::string& err) {
        const int64_t E = d.E, F = w.ffn_n_ff;
        tm.mark(kFfn, st);
        if (F != d.NFF_D) { err = "glm prefill: a dense FFN width differs from the geometry's"; return false; }
        if (w.ffn.gate_type != w.ffn.up_type || !sk::dequant_bf16_supported(w.ffn.gate_type) ||
            !sk::dequant_bf16_supported(w.ffn.down_type)) {
            err = "glm prefill: dense FFN weight types";
            return false;
        }
        // gate and up dequantized once, side by side: rows [0, F) gate, [F, 2F) up -> gu_d [S][2F]
        sk::dequant_f16(w.ffn.gate_type, w.ffn.gate, 0, F, E, b.wscr, st);
        sk::dequant_f16(w.ffn.up_type, w.ffn.up, 0, F, E, b.wscr + F * E, st);
        for (int64_t s0 = 0; s0 < T; s0 += kFfnSub) {
            const int64_t n = std::min(kFfnSub, T - s0);
            gemm.run(b.xn16 + s0 * E, E, b.wscr, b.gu_d, 2 * F, n, 2 * F, E);
            glm::swiglu_clamp_f16(b.gu_d, 2 * F, n, F, g->swiglu_clamp_shexp, b.h_d + s0 * F, st);
        }
        return proj(w.ffn.down, w.ffn.down_type, b.h_d, F, b.f, E, T, E, F, err);
    }

    bool dsa(int64_t l, const sc::GlmLayerWeights& w, int64_t T, int64_t p0, std::string& err) {
        const sk::DsaWeights& W = w.dsa;
        const sk::DsaShapes& s = v.dshape;
        const sk::DsaKvCache& c = v.dsa[(size_t) l];
        const int64_t E = d.E, QL = d.QL, KVL = d.KVL, DH = d.DH, HK = d.HK, HV = d.HV, IH = d.IH, ID = d.ID;
        tm.mark(kDsaProj, st);
        if (!proj(W.wq_a, W.wq_a_type, b.xn16, E, b.qa, QL, T, QL, E, err) ||
            !proj(W.wkv_a, W.wkv_a_type, b.xn16, E, b.kv, KVL, T, KVL, E, err) ||
            !proj(W.idx_k, W.idx_k_type, b.xn16, E, b.ikr, ID, T, ID, E, err) ||
            !proj(W.idx_gate, W.idx_gate_type, b.xn16, E, b.ig, ID, T, ID, E, err))
            return false;
        sk::dsa_gemv_f32(W.idx_proj, b.xn, b.iw, E, IH, T, 1.0f / sqrtf((float) (ID * IH)), st);
        sk::dsa_rms_norm(b.qa, W.q_a_norm, s.rms_eps, QL, T, b.qr, st);
        sk::dsa_rms_norm(b.kv, W.kv_a_norm, s.rms_eps, KVL, T, b.lat, st);
        sk::dsa_layer_norm(b.ikr, W.idx_k_norm, W.idx_k_norm_b, s.norm_eps, ID, T, b.ik, st);
        glm::to_f16(b.qr, b.qr16, T * QL, st);
        // the chunk's cells into the cache, then the pools it completes (pool j ends at cell kpool j + kpool - 1)
        tm.mark(kDsaCache, st);
        sk::dsa_kv_append_range(c, s, p0, T, b.lat, b.ik, b.ig, st);
        sk::dsa_pool_range(c, s, W.idx_ape, p0 / d.KP, (p0 + T) / d.KP, st);
        // the per-layer weights of the query side, dequantized once for the chunk's sub-batches
        if (!sk::dequant_bf16_supported(W.wq_b_type) || !sk::dequant_bf16_supported(W.idx_q_b_type)) {
            err = "glm prefill: wq_b / indexer.attn_q_b weight types";
            return false;
        }
        uint16_t* wq_b = b.dsaw;
        uint16_t* wiq = wq_b + DH * HK * QL;
        uint16_t* wk_b = wiq + IH * ID * QL;
        uint16_t* wv_b = wk_b + DH * KVL * HK;
        sk::dequant_f16(W.wq_b_type, W.wq_b, 0, DH * HK, QL, wq_b, st);
        sk::dequant_f16(W.idx_q_b_type, W.idx_q_b, 0, IH * ID, QL, wiq, st);
        sk::dequant_f16(8, W.wk_b, 0, DH * KVL, HK, wk_b, st);
        sk::dequant_f16(8, W.wv_b, 0, DH * HV, KVL, wv_b, st);
        const int64_t top = sk::dsa_top_pools(s);
        const float scale = 1.0f / sqrtf((float) HK);
        for (int64_t s0 = 0; s0 < T; s0 += kDsaSub) {
            const int64_t S = std::min(kDsaSub, T - s0);
            const int32_t* steps = b.steps + s0 * sk::kStepCount;
            tm.mark(kDsaQ, st);
            gemm.run(b.qr16 + s0 * QL, QL, wq_b, b.q, DH * HK, S, DH * HK, QL);
            glm::to_f16(b.q, b.q16, S * DH * HK, st);
            for (int64_t h = 0; h < DH; ++h)   // k_b absorbed: q_abs[t][h] = wk_b[h] q[t][h]
                gemm.run(b.q16 + h * HK, DH * HK, wk_b + h * KVL * HK, b.q_abs + h * KVL, DH * KVL, S, KVL, HK);
            // the selection: tokens whose complete pools exceed the top 512 need the indexer; the rest attend to
            // every cell up to themselves (the selection is the identity there)
            const int64_t pos_first = p0 + s0;
            int64_t sparse0 = S;   // first sub-batch row with (pos + 1) / kpool > top
            for (int64_t i = 0; i < S; ++i)
                if ((pos_first + i + 1) / d.KP > top) { sparse0 = i; break; }
            const int32_t* ids = nullptr;
            if (sparse0 < S) {
                tm.mark(kDsaIdx, st);
                gemm.run(b.qr16 + s0 * QL, QL, wiq, b.iq, IH * ID, S, IH * ID, QL);
                sk::dsa_index_scores(c, s, b.iq + sparse0 * IH * ID, b.iw + (s0 + sparse0) * IH,
                                     steps + sparse0 * sk::kStepCount, S - sparse0, d.max_pools,
                                     b.scores + sparse0 * d.max_pools, st);
                sk::dsa_select(b.scores, s, steps, S, d.max_pools, d.cap, b.ids, st);
                ids = b.ids;
            }
            tm.mark(kDsaAttn, st);
            glm::dsa_prompt_attend(c, s, b.q_abs, ids, d.cap, steps, S, scale, b.olat, st);
            tm.mark(kDsaOut, st);
            glm::to_f16(b.olat, b.olat16, S * DH * KVL, st);
            for (int64_t h = 0; h < DH; ++h)   // v_b: o[t][h] = wv_b[h] o_lat[t][h]
                gemm.run(b.olat16 + h * KVL, DH * KVL, wv_b + h * HV * KVL, b.ov + h * HV, DH * HV, S, HV, KVL);
            glm::to_f16(b.ov, b.o16 + s0 * DH * HV, S * DH * HV, st);
        }
        tm.mark(kDsaOut, st);
        return proj(W.wo, W.wo_type, b.o16, DH * HV, b.f, E, T, E, DH * HV, err);
    }

    bool moe(int64_t l, const sc::GlmLayerWeights& w, int64_t T, std::string& err) {
        const int64_t E = d.E, FF = d.FF_SH, K = d.K;
        const sk::NativeExpertLayout& L = lay[(size_t) l];
        if (L.bytes == 0 || src == nullptr) { err = "glm prefill: layer " + std::to_string(l) + " has no experts"; return false; }
        // the router (the decode kernels: F32 logits, sigmoid + bias selection, normalized and scaled weights)
        tm.mark(kRouter, st);
        sk::router_f32_logits(w.router, b.xn, b.logits, rs, (int) T, st);
        sk::router_sigmoid_topk(b.logits, w.exp_probs_b, b.rids, b.rw, rs, (int) T, st);
        cudaMemcpyAsync(h_ids, b.rids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(h_w, b.rw, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st);
        cudaEventRecord(ev_route, st);
        // the shared expert while the host waits for the ids
        tm.mark(kShared, st);
        if (w.ffn.gate_type != w.ffn.up_type) { err = "glm prefill: shared expert gate/up types differ"; return false; }
        if (!proj(w.ffn.gate, w.ffn.gate_type, b.xn16, E, b.gu, 2 * FF, T, FF, E, err) ||
            !proj(w.ffn.up, w.ffn.up_type, b.xn16, E, b.gu + FF, 2 * FF, T, FF, E, err))
            return false;
        glm::swiglu_clamp_f16(b.gu, 2 * FF, T, FF, g->swiglu_clamp_shexp, b.sh16, st);
        if (!proj(w.ffn.down, w.ffn.down_type, b.sh16, FF, b.shared, E, T, E, FF, err)) return false;
        cudaMemsetAsync(b.f, 0, (size_t) T * E * 4, st);
        const auto t_route = Clock::now();
        if (!cuda_ok(cudaEventSynchronize(ev_route), "router ids", err)) return false;
        stats.ms_route_sync += ms_since(t_route);
        // the plan: every entry (t, j) grouped by expert, experts in id order, tokens ascending within one
        std::vector<int32_t> count((size_t) d.NEXP + 1, 0);
        for (int64_t i = 0; i < T * K; ++i) {
            const int32_t e = h_ids[i];
            if (e < 0 || e >= d.NEXP) { err = "glm prefill: router id out of range"; return false; }
            ++count[(size_t) e + 1];
        }
        for (int64_t e = 0; e < d.NEXP; ++e) count[(size_t) e + 1] += count[(size_t) e];
        std::vector<int32_t> fill(count.begin(), count.end() - 1);
        for (int64_t i = 0; i < T * K; ++i) {
            const int32_t e = h_ids[i];
            const int32_t at = fill[(size_t) e]++;
            h_tok_sorted[at] = (int32_t) (i / K);
            h_w_sorted[at] = h_w[i];
        }
        cudaMemcpyAsync(b.tok_sorted, h_tok_sorted, (size_t) T * K * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(b.w_sorted, h_w_sorted, (size_t) T * K * 4, cudaMemcpyHostToDevice, st);
        ++stats.moe_calls;
        stats.entries += T * K;
        const glm::ExpertScratch xs{b.wscr, b.X16, b.GU, b.H16, b.D};
        // the experts, in id order
        for (int64_t e = 0; e < d.NEXP; ++e) {
            const int32_t a = count[(size_t) e], n = count[(size_t) e + 1] - a;
            if (n == 0) continue;
            ++stats.experts;
            const uint8_t* blob = nullptr;
            int k = -1;
            const int32_t slot = cache != nullptr ? cache->slot_of(l, e) : sc::kNotResident;
            if (slot != sc::kNotResident) {
                blob = cache->device_slot(slot);
                ++stats.vram;
            } else {
                k = (int) (ring_next++ % kRing);
                uint8_t* dev = b.ring + (uint64_t) k * ring_slot;
                uint8_t* host = h_ring + (uint64_t) k * ring_slot;
                if (ring_live[k]) {   // the slot's previous expert computed (and so its copy landed)
                    const auto t0 = Clock::now();
                    if (!cuda_ok(cudaEventSynchronize(ring_used[k]), "ring slot", err)) return false;
                    stats.ms_wait_slot += ms_since(t0);
                }
                const bool ram = src->has_resident(l, e);
                const auto t0 = Clock::now();
                if (src->pinned(l, e)) {   // page-locked RAM tier: DMA from its place
                    const uint8_t* hb = src->blob(l, e);
                    if (hb == nullptr) { err = "glm prefill: no blob for an expert"; return false; }
                    cudaMemcpyAsync(dev, hb, L.bytes, cudaMemcpyHostToDevice, copy);
                } else {                   // pageable RAM copy or the GGUF in place: through the staging slot
                    if (!src->copy_blob(l, e, host)) {
                        err = "glm prefill: reading layer " + std::to_string(l) + " expert " + std::to_string(e);
                        return false;
                    }
                    cudaMemcpyAsync(dev, host, L.bytes, cudaMemcpyHostToDevice, copy);
                }
                stats.ms_copy_host += ms_since(t0);
                cudaEventRecord(ring_copied[k], copy);
                tm.mark(kExpWait, st);
                cudaStreamWaitEvent(st, ring_copied[k], 0);
                blob = dev;
                stats.staged_bytes += L.bytes;
                ++(ram ? stats.ram : stats.file);
            }
            tm.mark(kExpGemm, st);
            glm::expert_rows(gemm, L, blob, b.xn, E, b.tok_sorted + a, b.w_sorted + a, n, g->swiglu_clamp_exp, xs,
                             b.f, st);
            if (k >= 0) {
                cudaEventRecord(ring_used[k], st);
                ring_live[k] = true;
            }
        }
        glm::add(b.f, b.shared, b.f, T * E, st);
        return true;
    }

    bool layer(int64_t l, int64_t T, int64_t p0, std::string& err) {
        const sc::GlmLayerWeights& w = (*v.w)[(size_t) l];
        bool ok = true;
        mhc_half(w, false, T, err, ok);
        if (!ok) return false;
        if (w.kind == sc::LayerKind::DsaMoe) ok = dsa(l, w, T, p0, err);
        else ok = kda(l, w, T, err);
        if (!ok) return false;
        tm.mark(kMhc, st);
        sk::mhc_post(b.f, b.x, b.post, b.comb, b.x, v.mshape, (int) T, st);
        mhc_half(w, true, T, err, ok);
        if (!ok) return false;
        if (w.kind == sc::LayerKind::KdaDense) ok = dense_ffn(w, T, err);
        else ok = moe(l, w, T, err);
        if (!ok) return false;
        tm.mark(kMhc, st);
        sk::mhc_post(b.f, b.x, b.post, b.comb, b.x, v.mshape, (int) T, st);
        if (sc::GlmtWriter* wr = dump[(size_t) l]) {
            const size_t n = (size_t) T * d.HC * d.E;
            h_dump.resize(n);
            if (!cuda_ok(cudaMemcpyAsync(h_dump.data(), b.x, n * 4, cudaMemcpyDeviceToHost, st), "dump", err) ||
                !cuda_ok(cudaStreamSynchronize(st), "dump", err))
                return false;
            if (!wr->append(h_dump.data(), T)) { err = "glm prefill: writing a dump"; return false; }
        }
        return true;
    }
};

GlmPrefill::GlmPrefill() : impl_(new Impl) {}
GlmPrefill::~GlmPrefill() = default;

uint64_t GlmPrefill::bytes_needed(const sc::ModelGeometry& g, const sc::GlmConfig& cfg, int64_t chunk) {
    return carve(dims_of(g, cfg, chunk), nullptr).bytes;
}

bool GlmPrefill::init(sc::GlmEngine& eng, sc::ExpertCache* cache, sc::FileExpertSource* src, int64_t chunk,
                      std::string& err) {
    Impl& m = *impl_;
    if (chunk < 1 || chunk > 32768) { err = "glm prefill: the chunk must be 1..32768 tokens"; return false; }
    m.eng = &eng;
    m.v = eng.session_view();
    m.g = m.v.g;
    m.cache = cache;
    m.src = src;
    m.st = (cudaStream_t) m.v.stream;
    const sc::ModelGeometry& g = *m.g;
    m.d = dims_of(g, *m.v.cfg, chunk);
    if (m.d.DI != m.d.NH * m.d.HD) { err = "glm prefill: d_inner != n_head * head_dim"; return false; }
    if (m.d.any_dsa && m.v.cfg->kv_fmt != sk::kDsaLatInt8 && m.v.cfg->kv_fmt != sk::kDsaLatF16) {
        err = "glm prefill: DSA latent format";
        return false;
    }
    const Bufs probe = carve(m.d, nullptr);
    if (!cuda_ok(cudaMalloc(&m.arena, probe.bytes), "the prompt buffers", err)) {
        cudaGetLastError();
        err += " (" + std::to_string(probe.bytes >> 20) + " MiB for chunks of " + std::to_string(chunk) + " tokens)";
        return false;
    }
    m.b = carve(m.d, (void*) (((uintptr_t) m.arena + kAlign - 1) / kAlign * kAlign));
    if (!m.gemm.init(m.st, err)) return false;
    if (!cuda_ok(cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking), "copy stream", err) ||
        !cuda_ok(cudaEventCreateWithFlags(&m.ev_route, cudaEventDisableTiming), "event", err))
        return false;
    for (int k = 0; k < kRing; ++k)
        if (!cuda_ok(cudaEventCreateWithFlags(&m.ring_copied[k], cudaEventDisableTiming), "event", err) ||
            !cuda_ok(cudaEventCreateWithFlags(&m.ring_used[k], cudaEventDisableTiming), "event", err))
            return false;
    const int64_t T = chunk;
    if (!cuda_ok(cudaHostAlloc((void**) &m.h_tok, (size_t) T * 4, cudaHostAllocDefault), "pinned", err) ||
        !cuda_ok(cudaHostAlloc((void**) &m.h_steps, (size_t) T * sk::kStepCount * 4, cudaHostAllocDefault), "pinned", err))
        return false;
    m.lay.assign((size_t) g.n_layers, sk::NativeExpertLayout{});
    if (m.d.any_moe) {
        if (src == nullptr) { err = "glm prefill: an MoE layer runs and no expert source was given"; return false; }
        const auto& lay = sk::cpu::expert_layout();
        if (!lay.native || lay.n_layers != g.n_layers) { err = "glm prefill: the native expert layout is not loaded"; return false; }
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!lay.has_experts(l)) continue;
            const auto& f = lay.fmt[(size_t) l];
            sk::NativeExpertLayout L = sk::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            if (L.bytes != lay.blob_bytes(l) || f.n_embd != g.n_embd || f.n_ff != g.n_ff ||
                (2 * f.n_ff * f.n_embd) % 256 != 0 || !sk::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                err = "glm prefill: layer " + std::to_string(l) + ": expert format " + std::to_string(f.gu_type) + "/" +
                      std::to_string(f.d_type) + " not taken by the FP16 dequantizer";
                return false;
            }
            m.lay[(size_t) l] = L;
        }
        m.rs.n_embd = (int) g.n_embd;
        m.rs.n_expert = (int) g.n_expert;
        m.rs.k = (int) g.n_expert_used;
        m.rs.norm = g.expert_weights_norm;
        m.rs.scale = g.expert_weights_scale;
        m.ring_slot = (m.d.max_blob + kAlign - 1) / kAlign * kAlign;
        if (!cuda_ok(cudaHostAlloc((void**) &m.h_ids, (size_t) T * m.d.K * 4, cudaHostAllocDefault), "pinned", err) ||
            !cuda_ok(cudaHostAlloc((void**) &m.h_w, (size_t) T * m.d.K * 4, cudaHostAllocDefault), "pinned", err) ||
            !cuda_ok(cudaHostAlloc((void**) &m.h_tok_sorted, (size_t) T * m.d.K * 4, cudaHostAllocDefault), "pinned", err) ||
            !cuda_ok(cudaHostAlloc((void**) &m.h_w_sorted, (size_t) T * m.d.K * 4, cudaHostAllocDefault), "pinned", err) ||
            !cuda_ok(cudaHostAlloc((void**) &m.h_ring, (size_t) kRing * m.ring_slot, cudaHostAllocDefault), "pinned", err))
            return false;
    }
    m.dump.assign((size_t) g.n_layers, nullptr);
    m.tm.on = std::getenv("STRATA_GLM_PREFILL_TIMING") != nullptr && std::getenv("STRATA_GLM_PREFILL_TIMING")[0] == '1';
    return true;
}

bool GlmPrefill::run(const int32_t* tokens, int64_t n, int64_t pos0, int n_logits, std::string& err) {
    Impl& m = *impl_;
    if (m.eng == nullptr) { err = "glm prefill: not initialized"; return false; }
    if (n <= 0) return true;
    const sc::ModelGeometry& g = *m.g;
    if (pos0 < 0 || pos0 + n > m.v.cfg->max_cells) {
        err = "glm prefill: positions " + std::to_string(pos0) + ".." + std::to_string(pos0 + n - 1) +
              " exceed the context (" + std::to_string(m.v.cfg->max_cells) + ")";
        return false;
    }
    const auto t0 = Clock::now();
    const int64_t C = m.d.T, E = m.d.E, HC = m.d.HC;
    for (int64_t q = 0; q < n; q += C) {
        const int64_t T = std::min(C, n - q), p0 = pos0 + q;
        m.tm.mark(kEmbed, m.st);
        for (int64_t t = 0; t < T; ++t) {
            const int32_t id = tokens[q + t];
            if (id < 0 || id >= g.n_vocab) { err = "glm prefill: token id out of range"; return false; }
            m.h_tok[t] = id;
        }
        // the previous chunk's uploads have landed (the stream was synchronized at its end)
        cudaMemcpyAsync(m.b.tok, m.h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, m.st);
        if (m.d.any_dsa) {
            sk::dsa_step_fill(m.h_steps, p0, T, m.v.dshape);
            cudaMemcpyAsync(m.b.steps, m.h_steps, (size_t) T * sk::kStepCount * 4, cudaMemcpyHostToDevice, m.st);
        }
        // hc_init: the embedding repeated into every stream
        m.v.embed->gather_dev(m.b.tok, T, m.b.f, m.st);
        const size_t row = (size_t) E * 4;
        for (int64_t h = 0; h < HC; ++h)
            cudaMemcpy2DAsync(m.b.x + h * E, row * (size_t) HC, m.b.f, row, row, (size_t) T, cudaMemcpyDeviceToDevice,
                              m.st);
        for (int64_t l = 0; l < m.v.n_run; ++l)
            if (!m.layer(l, T, p0, err)) return false;
        const bool last = q + T == n;
        if (last && m.eng->head_runs() && n_logits > 0) {
            const int nl = (int) std::min<int64_t>({(int64_t) n_logits, (int64_t) m.v.cfg->max_t, T});
            m.tm.mark(kHead, m.st);
            if (!m.eng->head_rows(m.b.x + (size_t) (T - nl) * HC * E, nl, p0 + T - nl, err)) return false;
        }
        m.tm.mark(kEmbed, m.st);
        if (!cuda_ok(cudaStreamSynchronize(m.st), "chunk", err) || !cuda_ok(cudaGetLastError(), "chunk kernels", err))
            return false;
        m.tm.fold();
        ++m.stats.chunks;
        m.stats.tokens += T;
    }
    m.stats.ms_total += ms_since(t0);
    return true;
}

void GlmPrefill::set_dump(int64_t layer, sc::GlmtWriter* w) {
    if (layer >= 0 && layer < (int64_t) impl_->dump.size()) impl_->dump[(size_t) layer] = w;
}

int64_t GlmPrefill::chunk() const { return impl_->d.T; }
uint64_t GlmPrefill::device_bytes() const { return impl_->b.bytes; }
const GlmPrefillStats& GlmPrefill::stats() const { return impl_->stats; }

std::string GlmPrefill::stats_line() const {
    const GlmPrefillStats& s = impl_->stats;
    char b[768];
    std::snprintf(b, sizeof b,
                  "glm prefill: %lld tokens in %lld chunks of up to %lld, %.1f ms, %.1f tok/s; %lld MoE passes, %lld "
                  "routed entries, %lld experts computed: %lld from VRAM, %lld from RAM, %lld from the file tier "
                  "(%.1f MB staged over PCIe; host: %.1f ms copying, %.1f ms waiting for a ring slot, %.1f ms for the "
                  "router ids)",
                  (long long) s.tokens, (long long) s.chunks, (long long) impl_->d.T, s.ms_total,
                  s.ms_total > 0 ? 1000.0 * (double) s.tokens / s.ms_total : 0.0, (long long) s.moe_calls,
                  (long long) s.entries, (long long) s.experts, (long long) s.vram, (long long) s.ram,
                  (long long) s.file, (double) s.staged_bytes / 1e6, s.ms_copy_host, s.ms_wait_slot, s.ms_route_sync);
    return b;
}

std::string GlmPrefill::timing_report() const {
    const Timer& t = impl_->tm;
    if (!t.on) return {};
    std::string s = "glm prefill GPU ms:";
    double total = 0;
    char b[64];
    for (int i = 0; i < kPhaseCount; ++i) {
        if (t.ms[i] <= 0) continue;
        std::snprintf(b, sizeof b, " %s %.1f", kPhaseNames[i], t.ms[i]);
        s += b;
        total += t.ms[i];
    }
    std::snprintf(b, sizeof b, " (total %.1f)", total);
    return s + b;
}

}  // namespace strata::prefill
