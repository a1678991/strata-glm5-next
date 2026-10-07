// src/core/glm_layer.cpp - GLM-5.3-Flash decode: the layers composed, the session, the window and its commit.
// See include/strata/core/glm.hpp for the order (llama.cpp f1cee99 src/models/glm5-next.cpp) and the window contract.
#include "strata/core/glm.hpp"

#include "strata/core/glm_experts.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::core {
namespace sk = strata::kernels;

namespace {

constexpr uint64_t kAlign = 256;

/// Lays regions out back to back (256-byte aligned) over `base`; with a null base it only counts.
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
    err = std::string("glm: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

sk::DsaShapes dsa_shapes(const ModelGeometry& g) {
    sk::DsaShapes s;
    s.n_embd = g.n_embd;
    s.n_head = g.n_head;
    s.q_lora = g.q_lora;
    s.kv_lora = g.kv_lora;
    s.head_k = g.head_dim - g.n_rot;
    s.head_v = g.head_dim_v;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    s.kpool = g.idx_kpool;
    s.idx_top_k = g.idx_top_k;
    s.rms_eps = g.norm_rms_eps;
    s.norm_eps = g.norm_eps;
    return s;
}

/// Every region of the session, in one place, so that the byte count and the carve cannot disagree.
struct Layout {
    struct Kda { float *hist, *state, *qkv, *h, *fb, *braw; };
    std::vector<Kda> kda;
    std::vector<uint8_t*> dsa_cache;
    uint8_t* dsa_scratch = nullptr;
    int32_t *tok = nullptr, *steps = nullptr, *nkeep = nullptr;
    float *emb, *x, *flat, *mixes, *post, *comb, *cur, *xn, *f;
    void *xq, *sq, *ffn_scratch;
    float *tmp3, *fa, *ga, *gb, *y;
    float *hm, *hn, *logits;
    uint64_t bytes = 0;
};

Layout carve(const ModelGeometry& g, const GlmConfig& cfg, void* base) {
    Layout L;
    Carver c{(uint8_t*) base, 0};
    const int64_t T = cfg.max_t, E = g.n_embd, HC = g.hc, DI = g.ssm_qk_dim, NH = g.ssm_v_heads,
                  HD = g.ssm_state_size, DC = g.ssm_d_conv;
    const sk::DsaShapes ds = dsa_shapes(g);
    L.kda.resize((size_t) g.n_layers);
    L.dsa_cache.assign((size_t) g.n_layers, nullptr);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (g.attn_layer(l)) {
            L.dsa_cache[(size_t) l] = c.take<uint8_t>(sk::dsa_kv_bytes(ds, cfg.kv_fmt, cfg.max_cells, 0));
            L.kda[(size_t) l] = {};
            continue;
        }
        auto& k = L.kda[(size_t) l];
        k.hist = c.take<float>((uint64_t) 3 * DI * (DC - 1));
        k.state = c.take<float>((uint64_t) NH * HD * HD);
        k.qkv = c.take<float>((uint64_t) T * 3 * DI);
        k.h = c.take<float>((uint64_t) T * 3 * DI);
        k.fb = c.take<float>((uint64_t) T * DI);
        k.braw = c.take<float>((uint64_t) T * NH);
    }
    if (g.n_qsa_layers() > 0)
        L.dsa_scratch = c.take<uint8_t>(sk::dsa_scratch_bytes(ds, T, sk::dsa_max_pools(cfg.max_cells, ds)));
    L.tok = c.take<int32_t>((uint64_t) T);
    L.steps = c.take<int32_t>((uint64_t) T * sk::kStepCount);
    L.nkeep = c.take<int32_t>(1);
    L.emb = c.take<float>((uint64_t) T * E);
    L.x = c.take<float>((uint64_t) T * HC * E);
    L.flat = c.take<float>((uint64_t) T * HC * E);
    L.mixes = c.take<float>((uint64_t) T * (2 + HC) * HC);
    L.post = c.take<float>((uint64_t) T * HC);
    L.comb = c.take<float>((uint64_t) T * HC * HC);
    L.cur = c.take<float>((uint64_t) T * E);
    L.xn = c.take<float>((uint64_t) T * E);
    L.f = c.take<float>((uint64_t) T * E);
    const int64_t widest = std::max<int64_t>({HC * E, DI, E, g.n_ff_dense, g.n_ff});
    L.xq = c.take<uint8_t>(sk::native_q8_1_bytes((int) widest, (int) T));
    L.sq = c.take<uint8_t>(sk::native_q8_1_bytes((int) std::max<int64_t>(DI, E), (int) T));
    L.tmp3 = c.take<float>((uint64_t) 3 * T * DI);
    L.fa = c.take<float>((uint64_t) T * HD);
    L.ga = c.take<float>((uint64_t) T * HD);
    L.gb = c.take<float>((uint64_t) T * DI);
    L.y = c.take<float>((uint64_t) T * DI);
    L.ffn_scratch = c.take<uint8_t>(sk::ffn_clamp_scratch_bytes((int) E, (int) std::max(g.n_ff_dense, g.n_ff), (int) T));
    L.hm = c.take<float>((uint64_t) T * E);
    L.hn = c.take<float>((uint64_t) T * E);
    L.logits = c.take<float>((uint64_t) T * g.n_vocab);
    L.bytes = c.used + kAlign;
    return L;
}

}  // namespace

// ================================ weights ================================

bool glm_bind_weights(const WeightTable& wt, const ModelGeometry& g, std::vector<GlmLayerWeights>& out,
                      std::string& err) {
    if (!g.glm()) { err = "glm: not a glm5-next geometry"; return false; }
    out.assign((size_t) g.n_layers, GlmLayerWeights{});
    for (int64_t l = 0; l < g.n_layers; ++l) {
        GlmLayerWeights& w = out[(size_t) l];
        w.layer = l;
        w.kind = layer_kind(g, l);
        const std::string pre = "blk." + std::to_string(l) + ".";
        bool ok = true;
        auto ref = [&](const char* suffix) -> const WeightRef* {
            const WeightRef* r = wt.find(pre + suffix);
            if (r == nullptr && ok) { err = "glm: the pack has no " + pre + suffix; ok = false; }
            return r;
        };
        auto q = [&](const char* suffix, const void*& p, int& type) {
            const WeightRef* r = ref(suffix);
            if (r == nullptr) return;
            if (r->native_data == nullptr) {
                if (ok) err = "glm: " + pre + suffix + " is not served natively (NativeDense did not attach it)";
                ok = false;
                return;
            }
            p = r->native_data;
            type = r->native_type;
        };
        auto f32 = [&](const char* suffix) -> const float* {
            const WeightRef* r = ref(suffix);
            if (r == nullptr) return nullptr;
            if (r->kind != WeightKind::F32 || r->data == nullptr) {
                if (ok) err = "glm: " + pre + suffix + " is not an F32 row in the arena";
                ok = false;
                return nullptr;
            }
            return (const float*) r->data;
        };
        q("hc_attn_fn.weight", w.hc_attn_fn, w.hc_attn_fn_type);
        w.hc_attn_scale = f32("hc_attn_scale.weight");
        w.hc_attn_base = f32("hc_attn_base.weight");
        q("hc_ffn_fn.weight", w.hc_ffn_fn, w.hc_ffn_fn_type);
        w.hc_ffn_scale = f32("hc_ffn_scale.weight");
        w.hc_ffn_base = f32("hc_ffn_base.weight");
        w.attn_norm = f32("attn_norm.weight");
        w.ffn_norm = f32("ffn_norm.weight");
        if (w.kind == LayerKind::DsaMoe) {
            sk::DsaWeights& d = w.dsa;
            q("attn_q_a.weight", d.wq_a, d.wq_a_type);
            d.q_a_norm = f32("attn_q_a_norm.weight");
            q("attn_q_b.weight", d.wq_b, d.wq_b_type);
            q("attn_kv_a_mqa.weight", d.wkv_a, d.wkv_a_type);
            d.kv_a_norm = f32("attn_kv_a_norm.weight");
            int kb = -1, vb = -1;
            q("attn_k_b.weight", d.wk_b, kb);
            q("attn_v_b.weight", d.wv_b, vb);
            if (ok && (kb != 8 || vb != 8)) {
                err = "glm: " + pre + "attn_k_b / attn_v_b must be Q8_0 (dsa_grouped_q8_0)";
                ok = false;
            }
            q("attn_output.weight", d.wo, d.wo_type);
            q("indexer.attn_q_b.weight", d.idx_q_b, d.idx_q_b_type);
            q("indexer.attn_k.weight", d.idx_k, d.idx_k_type);
            d.idx_k_norm = f32("indexer.k_norm.weight");
            d.idx_k_norm_b = f32("indexer.k_norm.bias");
            q("indexer_compressor_gate.weight", d.idx_gate, d.idx_gate_type);
            d.idx_ape = f32("indexer_compressor_ape.weight");
            d.idx_proj = f32("indexer.proj.weight");
        } else {
            q("attn_q.weight", w.wq, w.wq_type);
            q("attn_k.weight", w.wk, w.wk_type);
            q("attn_v.weight", w.wv, w.wv_type);
            w.conv_q = f32("ssm_conv1d_q.weight");
            w.conv_k = f32("ssm_conv1d_k.weight");
            w.conv_v = f32("ssm_conv1d_v.weight");
            q("ssm_f_a.weight", w.f_a, w.f_a_type);
            q("ssm_f_b.weight", w.f_b, w.f_b_type);
            q("ssm_g_a.weight", w.g_a, w.g_a_type);
            q("ssm_g_b.weight", w.g_b, w.g_b_type);
            q("ssm_beta.weight", w.beta, w.beta_type);
            w.dt_b = f32("ssm_dt.bias");
            w.ssm_a = f32("ssm_a");
            w.ssm_norm = f32("ssm_norm.weight");
            q("attn_output.weight", w.wo, w.wo_type);
        }
        if (const WeightRef* r = wt.find(pre + (w.kind == LayerKind::KdaDense ? "ffn_gate.weight" : "ffn_gate_shexp.weight")))
            w.ffn_n_ff = r->ne1;
        if (w.kind == LayerKind::KdaDense) {
            q("ffn_gate.weight", w.ffn.gate, w.ffn.gate_type);
            q("ffn_up.weight", w.ffn.up, w.ffn.up_type);
            q("ffn_down.weight", w.ffn.down, w.ffn.down_type);
        } else {
            q("ffn_gate_shexp.weight", w.ffn.gate, w.ffn.gate_type);
            q("ffn_up_shexp.weight", w.ffn.up, w.ffn.up_type);
            q("ffn_down_shexp.weight", w.ffn.down, w.ffn.down_type);
            w.router = f32("ffn_gate_inp.weight");
            w.exp_probs_b = f32("exp_probs_b.bias");
        }
        if (!ok) return false;
    }
    return true;
}

// ================================ GLMT ================================

bool GlmtWriter::create(const std::string& path, const std::string& name, int64_t ne0, int64_t ne1, std::string& err) {
    finish();
    f_ = std::fopen(path.c_str(), "wb");
    if (f_ == nullptr) { err = "glm: cannot write " + path; return false; }
    name_ = name;
    ne0_ = ne0;
    ne1_ = ne1;
    rows_ = 0;
    char hdr[128] = {};
    return std::fwrite(hdr, 1, sizeof hdr, f_) == sizeof hdr;   // patched by finish()
}

bool GlmtWriter::append(const float* rows, int64_t n_rows) {
    if (f_ == nullptr) return false;
    const size_t n = (size_t) (n_rows * ne0_ * ne1_);
    if (std::fwrite(rows, sizeof(float), n, f_) != n) return false;
    rows_ += n_rows;
    return true;
}

bool GlmtWriter::finish() {
    if (f_ == nullptr) return true;
    char hdr[128] = {};
    std::memcpy(hdr, "GLMT", 4);
    const uint32_t ver = 1, type = 0, nl = (uint32_t) std::min<size_t>(name_.size(), 63);
    const uint64_t row_elems = (uint64_t) (ne0_ * ne1_), rows = (uint64_t) rows_;
    const int64_t ne[4] = {ne0_, ne1_, rows_, 1};
    std::memcpy(hdr + 4, &ver, 4);
    std::memcpy(hdr + 8, &type, 4);
    std::memcpy(hdr + 12, &nl, 4);
    std::memcpy(hdr + 16, &row_elems, 8);
    std::memcpy(hdr + 24, &rows, 8);
    std::memcpy(hdr + 32, ne, 32);
    std::memcpy(hdr + 64, name_.data(), nl);
    std::fseek(f_, 0, SEEK_SET);
    const bool ok = std::fwrite(hdr, 1, sizeof hdr, f_) == sizeof hdr;
    const bool closed = std::fclose(f_) == 0;
    f_ = nullptr;
    return ok && closed;
}

// ================================ the session ================================

uint64_t glm_session_bytes(const ModelGeometry& g, const GlmConfig& cfg) {
    return g.glm() ? carve(g, cfg, nullptr).bytes : 0;
}

GlmEngine::~GlmEngine() {
    if (stream_) cudaStreamSynchronize((cudaStream_t) stream_);
    for (void* e : ev_) if (e) cudaEventDestroy((cudaEvent_t) e);
    if (h_dump_) cudaFreeHost(h_dump_);
    if (h_tok_) cudaFreeHost(h_tok_);
    if (h_steps_) cudaFreeHost(h_steps_);
    if (h_nkeep_) cudaFreeHost(h_nkeep_);
    if (arena_) cudaFree(arena_);
    if (stream_) cudaStreamDestroy((cudaStream_t) stream_);
}

bool GlmEngine::init(const WeightTable& wt, const ModelGeometry& g, const NativeEmbed& embed, const NativeHead* head,
                     GlmExperts* experts, const GlmConfig& cfg, std::string& err) {
    if (!g.glm()) { err = "glm: not a glm5-next geometry"; return false; }
    if (cfg.max_t < 1 || cfg.max_t > sk::DSA_MAX_T) { err = "glm: the window must be 1..8 tokens"; return false; }
    if (cfg.max_cells < 1 || cfg.max_cells > INT32_MAX / 2) { err = "glm: context out of range"; return false; }
    if (g.ssm_state_size != 128 || g.hc != 4 || !g.ssm_conv_separate || g.ssm_d_conv != 4) {
        err = "glm: the KDA/mHC kernels need head_dim 128, 4 streams, separate convs of 4 taps";
        return false;
    }
    g_ = g;
    cfg_ = cfg;
    embed_ = &embed;
    head_ = head;
    experts_ = experts;
    n_run_ = cfg.n_run < 0 || cfg.n_run > g.n_layers ? g.n_layers : cfg.n_run;
    if (!glm_bind_weights(wt, g, w_, err)) return false;
    output_norm_ = nullptr;
    if (const WeightRef* r = wt.find("output_norm.weight"); r != nullptr && r->kind == WeightKind::F32 && r->data)
        output_norm_ = (const float*) r->data;
    if (head_ != nullptr && output_norm_ == nullptr) { err = "glm: the pack has no F32 output_norm.weight"; return false; }

    kshape_.n_head = (int) g.ssm_v_heads;
    kshape_.head_dim = (int) g.ssm_state_size;
    kshape_.gate_lower_bound = g.kda_gate_lower_bound;
    kshape_.rms_eps = g.norm_rms_eps;
    kshape_.l2_eps = 1e-6f;   // G:738, "Match FLA l2 norm"
    mshape_.n_embd = (int) g.n_embd;
    mshape_.hc = (int) g.hc;
    mshape_.rms_eps = g.norm_rms_eps;
    mshape_.hc_eps = g.hc_eps;
    mshape_.sinkhorn_iters = (int) g.hc_sinkhorn_iters;
    dshape_ = dsa_shapes(g);

    cudaStream_t st = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream", err)) return false;
    stream_ = st;
    const Layout probe = carve(g, cfg, nullptr);
    arena_bytes_ = probe.bytes;
    if (!cuda_ok(cudaMalloc(&arena_, arena_bytes_), "the session arena", err)) return false;
    const Layout L = carve(g, cfg, (void*) (((uintptr_t) arena_ + kAlign - 1) / kAlign * kAlign));
    kda_.assign((size_t) g.n_layers, KdaLayer{});
    dsa_.assign((size_t) g.n_layers, sk::DsaKvCache{});
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (g.attn_layer(l)) {
            sk::dsa_kv_carve(L.dsa_cache[(size_t) l], dshape_, cfg.kv_fmt, cfg.max_cells, 0, dsa_[(size_t) l]);
        } else {
            const auto& k = L.kda[(size_t) l];
            kda_[(size_t) l] = KdaLayer{k.hist, k.state, k.qkv, k.h, k.fb, k.braw};
        }
    }
    if (L.dsa_scratch != nullptr)
        sk::dsa_scratch_carve(L.dsa_scratch, dshape_, cfg.max_t, sk::dsa_max_pools(cfg.max_cells, dshape_), dsc_);
    tok_ = L.tok; steps_ = L.steps; nkeep_ = L.nkeep;
    emb_ = L.emb; x_ = L.x; flat_ = L.flat; mixes_ = L.mixes; post_ = L.post; comb_ = L.comb;
    cur_ = L.cur; xn_ = L.xn; f_ = L.f; xq_ = L.xq; sq_ = L.sq; ffn_scratch_ = L.ffn_scratch;
    tmp3_ = L.tmp3; fa_ = L.fa; ga_ = L.ga; gb_ = L.gb; y_ = L.y;
    hm_ = L.hm; hn_ = L.hn; logits_ = L.logits;
    if (!cuda_ok(cudaHostAlloc((void**) &h_tok_, (size_t) cfg.max_t * 4, cudaHostAllocDefault), "staging", err) ||
        !cuda_ok(cudaHostAlloc((void**) &h_steps_, (size_t) cfg.max_t * sk::kStepCount * 4, cudaHostAllocDefault),
                 "staging", err) ||
        !cuda_ok(cudaHostAlloc((void**) &h_nkeep_, 4, cudaHostAllocDefault), "staging", err) ||
        !cuda_ok(cudaHostAlloc((void**) &h_dump_, (size_t) cfg.max_t * g.hc * g.n_embd * 4, cudaHostAllocDefault),
                 "staging", err))
        return false;
    dump_.assign((size_t) g.n_layers, nullptr);
    prof_ = std::getenv("STRATA_GLM_PROFILE") != nullptr && std::getenv("STRATA_GLM_PROFILE")[0] == '1';
    if (prof_) {
        ev_.assign((size_t) (g.n_layers + 2), nullptr);
        for (auto& e : ev_) {
            cudaEvent_t ce = nullptr;
            if (!cuda_ok(cudaEventCreate(&ce), "event", err)) return false;
            e = ce;
        }
        prof_ms_.assign((size_t) (g.n_layers + 1), 0.0);
    }
    for (int64_t l = 0; l < n_run_; ++l)
        if (layer_kind(g, l) != LayerKind::KdaDense && experts_ == nullptr) {
            err = "glm: layer " + std::to_string(l) + " is an MoE layer and no expert tier was given";
            return false;
        }
    return reset(err);
}

bool GlmEngine::reset(std::string& err) {
    cudaStream_t st = (cudaStream_t) stream_;
    for (int64_t l = 0; l < g_.n_layers; ++l) {
        if (g_.attn_layer(l)) {
            sk::dsa_kv_reset(dsa_[(size_t) l], dshape_, st);
        } else {
            const KdaLayer& k = kda_[(size_t) l];
            cudaMemsetAsync(k.hist, 0, (size_t) 3 * g_.ssm_qk_dim * (g_.ssm_d_conv - 1) * 4, st);
            cudaMemsetAsync(k.state, 0, (size_t) g_.ssm_v_heads * g_.ssm_state_size * g_.ssm_state_size * 4, st);
        }
    }
    last_t_ = 0;
    return cuda_ok(cudaStreamSynchronize(st), "reset", err);
}

void GlmEngine::set_dump(int64_t layer, GlmtWriter* w) {
    if (layer >= 0 && layer < (int64_t) dump_.size()) dump_[(size_t) layer] = w;
}

namespace {
// the debugging nodes, in graph order (llama.cpp's names; "#0" / "#1": the attention / FFN half's hc_mixes)
const char* const kNodes[] = {"hc_mixes#0", "hc_attn_pre", "attn_norm", "mixer_out", "hc_attn_post",
                              "hc_mixes#1", "hc_ffn_pre", "ffn_norm", "ffn_out"};
}  // namespace

bool GlmEngine::set_node_dump(int64_t layer, const std::string& dir, std::string& err) {
    node_layer_ = layer;
    nodes_.clear();
    const bool dsa = g_.attn_layer(layer);
    for (const char* n : kNodes) {
        std::string name = n;
        if (name == "mixer_out") name = dsa ? "attn_out" : "kda_out";
        const bool streams = name == "hc_attn_post";
        const int64_t ne0 = name.rfind("hc_mixes", 0) == 0 ? (2 + g_.hc) * g_.hc : g_.n_embd;
        auto w = std::make_unique<GlmtWriter>();
        std::string file = name;
        for (char& c : file) if (c == '#') c = '_';
        const std::string full = name.substr(0, name.find('#')) + "-" + std::to_string(layer) +
                                 (name.find('#') != std::string::npos ? name.substr(name.find('#')) : "");
        if (!w->create(dir + "/" + file + "-" + std::to_string(layer) + ".glmt", full, ne0, streams ? g_.hc : 1, err))
            return false;
        nodes_.push_back(std::move(w));
    }
    h_node_.resize((size_t) cfg_.max_t * g_.hc * g_.n_embd);
    return true;
}

bool GlmEngine::dump_node(int i, const float* dev, int64_t per_token, int T, std::string& err) {
    if (nodes_.empty()) return true;
    cudaStream_t st = (cudaStream_t) stream_;
    cudaMemcpyAsync(h_node_.data(), dev, (size_t) T * per_token * 4, cudaMemcpyDeviceToHost, st);
    if (!cuda_ok(cudaStreamSynchronize(st), "node dump", err)) return false;
    if (!nodes_[(size_t) i]->append(h_node_.data(), T)) { err = "glm: writing a node dump failed"; return false; }
    return true;
}

// ================================ one window ================================

bool GlmEngine::mhc_half(const GlmLayerWeights& w, bool ffn, int T) {
    cudaStream_t st = (cudaStream_t) stream_;
    const int hcd = (int) (g_.hc * g_.n_embd), mix = mshape_.mix_dim();
    sk::mhc_flat_norm(x_, flat_, mshape_, T, st);
    sk::native_quantize_q8_1(flat_, xq_, hcd, T, st);
    sk::native_mmvq(ffn ? w.hc_ffn_fn_type : w.hc_attn_fn_type, ffn ? w.hc_ffn_fn : w.hc_attn_fn, xq_, mixes_, hcd, mix,
                    T, st);
    sk::mhc_pre(x_, mixes_, ffn ? w.hc_ffn_scale : w.hc_attn_scale, ffn ? w.hc_ffn_base : w.hc_attn_base, cur_, post_,
                comb_, mshape_, T, st);
    sk::dsa_rms_norm(cur_, ffn ? w.ffn_norm : w.attn_norm, g_.norm_rms_eps, g_.n_embd, T, xn_, st);
    return true;
}

bool GlmEngine::kda(int64_t l, const GlmLayerWeights& w, int T) {
    cudaStream_t st = (cudaStream_t) stream_;
    const KdaLayer& k = kda_[(size_t) l];
    const int E = (int) g_.n_embd, DI = (int) g_.ssm_qk_dim, HD = (int) g_.ssm_state_size, NH = (int) g_.ssm_v_heads;
    // the three projections share one Q8_1 image of xn; each writes [T][d_inner] into tmp3, then q|k|v per token
    sk::native_quantize_q8_1(xn_, xq_, E, T, st);
    sk::native_mmvq(w.wq_type, w.wq, xq_, tmp3_, E, DI, T, st);
    sk::native_mmvq(w.wk_type, w.wk, xq_, tmp3_ + (size_t) T * DI, E, DI, T, st);
    sk::native_mmvq(w.wv_type, w.wv, xq_, tmp3_ + (size_t) 2 * T * DI, E, DI, T, st);
    for (int p = 0; p < 3; ++p)
        cudaMemcpy2DAsync(k.qkv + (size_t) p * DI, (size_t) 3 * DI * 4, tmp3_ + (size_t) p * T * DI, (size_t) DI * 4,
                          (size_t) DI * 4, (size_t) T, cudaMemcpyDeviceToDevice, st);
    sk::native_mmvq(w.f_a_type, w.f_a, xq_, fa_, E, HD, T, st);
    sk::native_mmvq(w.g_a_type, w.g_a, xq_, ga_, E, HD, T, st);
    sk::native_mmvq(w.beta_type, w.beta, xq_, k.braw, E, NH, T, st);
    sk::native_quantize_q8_1(fa_, sq_, HD, T, st);
    sk::native_mmvq(w.f_b_type, w.f_b, sq_, k.fb, HD, DI, T, st);
    sk::native_quantize_q8_1(ga_, sq_, HD, T, st);
    sk::native_mmvq(w.g_b_type, w.g_b, sq_, gb_, HD, DI, T, st);
    // the verify half: history and state are read, never written (commit() replays the kept tokens)
    sk::kda_conv_l2(k.hist, k.qkv, w.conv_q, w.conv_k, w.conv_v, k.h, kshape_, T, false, st);
    sk::kda_step_norm(k.state, k.h, k.fb, w.dt_b, w.ssm_a, k.braw, gb_, w.ssm_norm, y_, kshape_, T, false, st);
    sk::native_quantize_q8_1(y_, sq_, DI, T, st);
    sk::native_mmvq(w.wo_type, w.wo, sq_, f_, DI, E, T, st);
    return true;
}

bool GlmEngine::layer(int64_t l, int T, std::string& err) {
    cudaStream_t st = (cudaStream_t) stream_;
    const GlmLayerWeights& w = w_[(size_t) l];
    const bool dbg = l == node_layer_ && !nodes_.empty();
    const int64_t E = g_.n_embd, MIX = (2 + g_.hc) * g_.hc;
    // the attention half
    mhc_half(w, false, T);
    if (dbg && !(dump_node(0, mixes_, MIX, T, err) && dump_node(1, cur_, E, T, err) && dump_node(2, xn_, E, T, err)))
        return false;
    if (w.kind == LayerKind::DsaMoe) sk::dsa_decode(w.dsa, dshape_, dsa_[(size_t) l], xn_, steps_, T, dsc_, f_, st);
    else kda(l, w, T);
    if (dbg && !dump_node(3, f_, E, T, err)) return false;
    sk::mhc_post(f_, x_, post_, comb_, x_, mshape_, T, st);
    if (dbg && !dump_node(4, x_, g_.hc * E, T, err)) return false;
    // the FFN half
    mhc_half(w, true, T);
    if (dbg && !(dump_node(5, mixes_, MIX, T, err) && dump_node(6, cur_, E, T, err) && dump_node(7, xn_, E, T, err)))
        return false;
    if (w.kind == LayerKind::KdaDense) {
        sk::ffn_clamp(w.ffn, xn_, f_, (int) g_.n_embd, (int) w.ffn_n_ff, T, g_.swiglu_clamp_shexp, ffn_scratch_, st);
    } else if (!experts_->moe(w, xn_, T, f_, st, err)) {
        return false;
    }
    if (dbg && !dump_node(8, f_, E, T, err)) return false;
    sk::mhc_post(f_, x_, post_, comb_, x_, mshape_, T, st);
    return true;
}

bool GlmEngine::head(int T, std::string& err) {
    (void) err;
    cudaStream_t st = (cudaStream_t) stream_;
    sk::mhc_mean(x_, hm_, mshape_, T, st);
    sk::dsa_rms_norm(hm_, output_norm_, g_.norm_rms_eps, g_.n_embd, T, hn_, st);
    sk::native_quantize_q8_1(hn_, xq_, (int) g_.n_embd, T, st);
    sk::native_mmvq(head_->type(), head_->weights(), xq_, logits_, (int) g_.n_embd, (int) g_.n_vocab, T, st);
    return true;
}

bool GlmEngine::run(int T, const int32_t* tokens, int64_t pos0, std::string& err, const float* x_in, int64_t from) {
    if (T < 1 || T > cfg_.max_t) { err = "glm: window of " + std::to_string(T) + " tokens"; return false; }
    if (pos0 < 0 || pos0 + T > cfg_.max_cells) {
        err = "glm: positions " + std::to_string(pos0) + ".." + std::to_string(pos0 + T - 1) + " exceed the context (" +
              std::to_string(cfg_.max_cells) + ")";
        return false;
    }
    cudaStream_t st = (cudaStream_t) stream_;
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= g_.n_vocab) { err = "glm: token id out of range"; return false; }
        h_tok_[t] = tokens[t];
    }
    cudaMemcpyAsync(tok_, h_tok_, (size_t) T * 4, cudaMemcpyHostToDevice, st);
    if (g_.n_qsa_layers() > 0) {
        sk::dsa_step_fill(h_steps_, pos0, T, dshape_);
        cudaMemcpyAsync(steps_, h_steps_, (size_t) T * sk::kStepCount * 4, cudaMemcpyHostToDevice, st);
    }
    if (x_in != nullptr) {   // the reference's streams entering layer `from`
        if (from < 0 || from >= n_run_) { err = "glm: input layer out of range"; return false; }
        cudaMemcpyAsync(x_, x_in, (size_t) T * g_.hc * g_.n_embd * 4, cudaMemcpyHostToDevice, st);
    } else {
        from = 0;
        // hc_init: the embedding repeated into every stream
        embed_->gather_dev(tok_, T, emb_, st);
        const size_t row = (size_t) g_.n_embd * 4;
        for (int64_t h = 0; h < g_.hc; ++h)
            cudaMemcpy2DAsync(x_ + h * g_.n_embd, row * (size_t) g_.hc, emb_, row, row, (size_t) T,
                              cudaMemcpyDeviceToDevice, st);
    }
    for (int64_t l = 0; l < from; ++l)
        if (prof_) cudaEventRecord((cudaEvent_t) ev_[(size_t) l], st);
    for (int64_t l = from; l < n_run_; ++l) {
        if (prof_) cudaEventRecord((cudaEvent_t) ev_[(size_t) l], st);
        if (!layer(l, T, err)) return false;
        if (GlmtWriter* w = dump_[(size_t) l]) {
            const size_t n = (size_t) T * g_.hc * g_.n_embd;
            cudaMemcpyAsync(h_dump_, x_, n * 4, cudaMemcpyDeviceToHost, st);
            if (!cuda_ok(cudaStreamSynchronize(st), "dump", err)) return false;
            if (!w->append(h_dump_, T)) { err = "glm: writing the l_out dump failed"; return false; }
        }
    }
    if (prof_) cudaEventRecord((cudaEvent_t) ev_[(size_t) n_run_], st);
    if (head_runs() && !head(T, err)) return false;
    if (prof_) cudaEventRecord((cudaEvent_t) ev_[(size_t) n_run_ + 1], st);
    if (!cuda_ok(cudaStreamSynchronize(st), "window", err)) return false;
    if (!cuda_ok(cudaGetLastError(), "window kernels", err)) return false;
    if (prof_) {
        for (int64_t l = 0; l <= n_run_; ++l) {
            float ms = 0;
            cudaEventElapsedTime(&ms, (cudaEvent_t) ev_[(size_t) l], (cudaEvent_t) ev_[(size_t) l + 1]);
            prof_ms_[(size_t) (l == n_run_ ? g_.n_layers : l)] += ms;
        }
        ++prof_windows_;
    }
    last_t_ = T;
    last_pos0_ = pos0;
    return true;
}

bool GlmEngine::commit(int n_keep, std::string& err) {
    if (n_keep < 1 || n_keep > last_t_) { err = "glm: commit count out of range"; return false; }
    cudaStream_t st = (cudaStream_t) stream_;
    h_nkeep_[0] = n_keep;
    cudaMemcpyAsync(nkeep_, h_nkeep_, 4, cudaMemcpyHostToDevice, st);
    for (int64_t l = 0; l < n_run_; ++l) {
        if (g_.attn_layer(l)) continue;
        const KdaLayer& k = kda_[(size_t) l];
        const GlmLayerWeights& w = w_[(size_t) l];
        sk::kda_conv_commit(k.hist, k.qkv, kshape_, nkeep_, st);
        sk::kda_step_commit(k.state, k.h, k.fb, w.dt_b, w.ssm_a, k.braw, kshape_, nkeep_, st);
    }
    last_t_ = 0;
    return cuda_ok(cudaStreamSynchronize(st), "commit", err);
}

bool GlmEngine::copy_logits(int t, float* host) const {
    if (!head_runs() || t < 0 || t >= cfg_.max_t) return false;
    return cudaMemcpy(host, logits_ + (size_t) t * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost) ==
           cudaSuccess;
}

bool GlmEngine::argmax(int T, int32_t* out) const {
    std::vector<float> row((size_t) g_.n_vocab);
    for (int t = 0; t < T; ++t) {
        if (!copy_logits(t, row.data())) return false;
        out[t] = (int32_t) (std::max_element(row.begin(), row.end()) - row.begin());
    }
    return true;
}

bool GlmEngine::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                                std::string& err) const {
    // Verifier::window_logprobs (src/core/verify.cpp), the same columns and arithmetic, so a tool that
    // reads Qwen's STRATA_LOGPOS files reads these alike
    const int64_t n_vocab = g_.n_vocab;
    if (!head_runs() || T <= 0 || targets == nullptr || out == nullptr) {
        err = "glm: window_logprobs: no head logits for this window";
        return false;
    }
    std::vector<float> h((size_t) T * (size_t) n_vocab);
    if (cudaMemcpy(h.data(), logits_, h.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "glm: window_logprobs: the head logits copy failed";
        return false;
    }
    static const int topk = [] {
        const char* v = std::getenv("STRATA_LOGPOS_TOPK");
        return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
    }();
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab;
        double sum = 0.0, sum_without = 0.0;
        for (int64_t v = 0; v < n_vocab; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab);
            for (int64_t v = 0; v < n_vocab; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
}

std::string GlmEngine::profile_report() const {
    if (!prof_ || prof_windows_ == 0) return {};
    std::string s = "glm profile (ms per window, " + std::to_string(prof_windows_) + " windows):";
    double total = 0;
    char b[64];
    for (int64_t l = 0; l < n_run_; ++l) {
        std::snprintf(b, sizeof b, " L%lld %.3f", (long long) l, prof_ms_[(size_t) l] / (double) prof_windows_);
        s += b;
        total += prof_ms_[(size_t) l];
    }
    std::snprintf(b, sizeof b, " head %.3f total %.3f", prof_ms_[(size_t) g_.n_layers] / (double) prof_windows_,
                  (total + prof_ms_[(size_t) g_.n_layers]) / (double) prof_windows_);
    return s + b;
}

}  // namespace strata::core
