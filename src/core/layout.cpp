// src/core/layout.cpp - the shape checks.  See the header for why this exists.
#include "strata/core/layout.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace strata::core {
namespace {

/// A required 2-D shape.  `ne0` is the CONTIGUOUS axis, matching the manifest and `s_gemv`'s convention
/// (`y[o] = sum_i x[i]*W[i][o]`, row o contiguous of length ne0).
struct Want2 {
    const char* suffix;
    int64_t ne0;
    int64_t ne1;
    bool qsa_only;
    bool gdn_only;
};

/// A required element count and ENGINE FORM for a 1-D tensor.
///
/// `kind` is not decoration.  The engine form of a 1-D tensor is whatever `WeightKind` the loader applied -
/// `F32` copies 4 B/elem, `Bf16InF32` re-rounds to 2 - and a consumer that casts either one to `const float*`
/// reads 2x its length.  **THAT IS NOT HYPOTHETICAL: `ffn_gate_inp_shexp.weight` is BF16, so the arena holds
/// 5120 B for 2560 elements, and `shared_expert` read it as f32.  Every one of the 2560 MoE outputs came out
/// non-finite** - a wrong answer loud enough to notice, which is the lucky version.  Reading 5120 B past the
/// end of a tensor inside a 4.5 GiB arena does not fault, so the same mistake on a tensor followed by
/// plausible bytes would have produced plausible logits.
struct Want1 {
    const char* suffix;
    int64_t elements;
    WeightKind kind;
    bool qsa_only;
    bool gdn_only;
};

/// A layer's tensor names, `blk.<layer>.<suffix>` (LayerView's spelling, without a table).
struct Names {
    int64_t l;
    int64_t layer() const { return l; }
    std::string name(const char* suffix) const { return "blk." + std::to_string(l) + "." + suffix; }
};

bool fail(std::string& err, const Names& v, const char* suffix, const char* what, int64_t got,
          int64_t want) {
    char buf[512];
    std::snprintf(buf, sizeof buf, "layer %lld: %s %s is %lld, the kernels require %lld",
                  (long long) v.layer(), v.name(suffix).c_str(), what, (long long) got, (long long) want);
    err = buf;
    return false;
}

/// What a lookup found for one tensor name.  A pack (`WeightTable`) knows two dimensions and the engine form; a
/// model file knows every dimension and no engine form.
struct Found {
    bool present = false;
    int nd = 0;                    ///< dimensions known exactly: 2 for a pack row, the GGUF's own count for a file
    int64_t ne[4] = {0, 0, 0, 0};
    int64_t elements = 0;
    bool has_kind = false;         ///< a pack row carries the loader's WeightKind; a GGUF tensor does not
    WeightKind kind = WeightKind::Verbatim;
    uint64_t bytes = 0;
};

/// A pack's table, as the Qwen checks have always read it.
struct TableLookup {
    const WeightTable* t;
    Found operator()(const std::string& name) const {
        Found f;
        const WeightRef* r = t->find(name);
        if (!r) return f;
        f.present = true;
        f.nd = 2;
        f.ne[0] = r->ne0;
        f.ne[1] = r->ne1;
        f.elements = r->elements;
        f.has_kind = true;
        f.kind = r->kind;
        f.bytes = r->bytes;
        return f;
    }
};

/// A model file's tensor directory (headers only).  Trailing dimensions of 1 are dropped, as llama.cpp accepts
/// `{4, 1, 8192, 1}` for `{4, 1, 8192}`.
struct GgufLookup {
    const strata::GgufModel* m;
    Found operator()(const std::string& name) const {
        Found f;
        const strata::TensorInfo* t = m->find(name);
        if (!t) return f;
        f.present = true;
        int nd = (int) std::min<size_t>(t->shape.size(), 4);
        while (nd > 1 && t->shape[(size_t) nd - 1] == 1) --nd;
        f.nd = nd;
        f.elements = 1;
        for (int d = 0; d < (int) t->shape.size(); ++d) {
            if (d < 4) f.ne[d] = (int64_t) t->shape[(size_t) d];
            f.elements *= (int64_t) t->shape[(size_t) d];
        }
        f.bytes = strata::tensor_payload_bytes(*t);
        return f;
    }
};

template <class Lookup>
bool check_one(const Lookup& lk, const ModelGeometry& g, int64_t layer, std::string& err) {
    const Names v{layer};
    const bool qsa = is_qsa_layer(g, layer);

    // ---- the 2-D tensor set, per layer family
    const Want2 want2[] = {
        // gated residual, EVERY layer - `gr_read` takes w_down (hc_lr, hc_dim) and w_up (hc_lr, hc_dim)
        // after its own transpose, so the pack's orientation is [hc_dim, hc_lr] and [hc_lr, hc_dim].
        {"hc_attn_down.weight", g.hc_dim(), g.hc_lr, false, false},
        {"hc_attn_up.weight", g.hc_lr, g.hc_dim(), false, false},
        {"hc_attn_inject.weight", g.hc_dim(), g.hc, false, false},
        {"hc_ffn_down.weight", g.hc_dim(), g.hc_lr, false, false},
        {"hc_ffn_up.weight", g.hc_lr, g.hc_dim(), false, false},
        {"hc_ffn_inject.weight", g.hc_dim(), g.hc, false, false},
        // MoE, EVERY layer
        {"ffn_gate_inp.weight", g.n_embd, g.n_expert, false, false},
        {"ffn_gate_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_up_shexp.weight", g.n_embd, g.n_ff, false, false},
        {"ffn_down_shexp.weight", g.n_ff, g.n_embd, false, false},
        // GDN only
        {"attn_qkv.weight", g.n_embd, g.ssm_conv_channels, false, true},
        {"attn_gate.weight", g.n_embd, g.ssm_value_dim, false, true},
        {"ssm_out.weight", g.ssm_value_dim, g.n_embd, false, true},
        {"ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels, false, true},
        {"ssm_alpha.weight", g.n_embd, g.ssm_v_heads, false, true},
        {"ssm_beta.weight", g.n_embd, g.ssm_v_heads, false, true},
        // QSA only
        {"attn_q.weight", g.n_embd, 2 * g.n_head * g.head_dim, true, false},
        {"attn_k.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_v.weight", g.n_embd, g.n_head_kv * g.head_dim, true, false},
        {"attn_output.weight", g.n_head * g.head_dim, g.n_embd, true, false},
        // THE INDEXER.  `q_proj` is the QUERY count and `k_proj` is the KEY width, and they are different
        // numbers - conflating them is what made the planner's indexer term 4x too big in round 194.
        {"indexer.q_proj.weight", g.n_embd, g.idx_q_heads * g.idx_key_dim, true, false},
        {"indexer.k_proj.weight", g.n_embd, g.idx_key_dim, true, false},
    };
    for (const Want2& w : want2) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const Found r = lk(v.name(w.suffix));
        if (!r.present) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r.ne[0] != w.ne0) return fail(err, v, w.suffix, "ne0", r.ne[0], w.ne0);
        if (r.ne[1] != w.ne1) return fail(err, v, w.suffix, "ne1", r.ne[1], w.ne1);
    }

    // ---- the 1-D set.  EVERY ONE OF THESE IS `F32` EXCEPT the shared expert's scalar gate, which is BF16 -
    // and that single exception is the one a reader would not guess, so it is written down rather than
    // inferred from the tensor count.  (Engine forms are a pack's: a model file's tensors are checked by count.)
    const Want1 want1[] = {
        {"hc_attn_norm.weight", g.hc_dim(), WeightKind::F32, false, false},
        {"hc_ffn_norm.weight", g.hc_dim(), WeightKind::F32, false, false},
        {"ffn_gate_inp_shexp.weight", g.n_embd, WeightKind::Bf16InF32, false, false},
        {"ssm_a", g.ssm_v_heads, WeightKind::F32, false, true},
        {"ssm_dt.bias", g.ssm_v_heads, WeightKind::F32, false, true},
        {"ssm_norm.weight", g.ssm_state_size, WeightKind::F32, false, true},
        {"attn_q_norm.weight", g.head_dim, WeightKind::F32, true, false},
        {"attn_k_norm.weight", g.head_dim, WeightKind::F32, true, false},
        {"indexer.q_norm.weight", g.idx_key_dim, WeightKind::F32, true, false},
        {"indexer.k_norm.weight", g.idx_key_dim, WeightKind::F32, true, false},
    };
    for (const Want1& w : want1) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;
        const Found r = lk(v.name(w.suffix));
        if (!r.present) {
            err = "layer " + std::to_string(layer) + ": missing " + v.name(w.suffix);
            return false;
        }
        if (r.elements != w.elements) return fail(err, v, w.suffix, "elements", r.elements, w.elements);
        if (r.has_kind && r.kind != w.kind) {
            // The byte count is what makes this a REAL check and not a label: a 4 B/elem tensor holds
            // `elements * 4`, a 2 B/elem one holds `elements * 2`, and a consumer reading it as the wrong one
            // walks off the end.
            const uint64_t want_bytes = (uint64_t) w.elements *
                                        (w.kind == WeightKind::Bf16InF32 ? 2u : 4u);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, v.name(w.suffix).c_str(), (int) r.kind,
                          (unsigned long long) r.bytes, (int) w.kind, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
    }
    return true;
}

// ---- glm5-next.  The names and shapes are llama.cpp f1cee99's `llama_model_glm5_next::load_arch_tensors`
// (src/models/glm5-next.cpp:62-185), checked against the UD-IQ4_XS file's tensor directory.

/// Which layers a tensor belongs to.
enum GlmWhen : unsigned {
    kNorm = 1u << 0,    ///< every block, the MTP block included: attn_norm, ffn_norm
    kHc = 1u << 1,      ///< every trunk block: the mHC mixing tensors
    kKda = 1u << 2,
    kDsa = 1u << 3,
    kDense = 1u << 4,   ///< the leading dense FFN
    kMoe = 1u << 5,     ///< router, bias, shared expert
    kExperts = 1u << 6, ///< the routed experts: a model file only (a pack keeps them in experts.bin)
    kMtp = 1u << 7,     ///< the NextN block's own tensors
};

/// A required shape: `nd` dimensions, ne0 contiguous.  A 1-D tensor is checked by its element count.
struct WantN {
    const char* suffix;
    int nd;
    int64_t ne[3];
    unsigned when;
    bool optional = false;   ///< checked when present (llama.cpp's TENSOR_NOT_REQUIRED)
};

std::vector<WantN> glm_wants(const ModelGeometry& g) {
    const int64_t E = g.n_embd, KD = g.ssm_qk_dim, VD = g.ssm_value_dim, HD = g.ssm_state_size;
    const int64_t mix = (2 + g.hc) * g.hc;   // pre (hc) + post (hc) + comb (hc x hc)
    const int64_t shexp = g.n_ff * g.n_expert_shared;
    return {
        {"attn_norm.weight", 1, {E}, kNorm},
        {"ffn_norm.weight", 1, {E}, kNorm},
        {"hc_attn_fn.weight", 2, {g.hc_dim(), mix}, kHc},
        {"hc_attn_base.weight", 1, {mix}, kHc},
        {"hc_attn_scale.weight", 1, {3}, kHc},
        {"hc_ffn_fn.weight", 2, {g.hc_dim(), mix}, kHc},
        {"hc_ffn_base.weight", 1, {mix}, kHc},
        {"hc_ffn_scale.weight", 1, {3}, kHc},
        // KDA: separate q, k, v projections and causal convs, the low-rank decay (f_a, f_b) and output gate
        // (g_a, g_b), beta per head, A per head, dt bias per channel, the output RMSNorm over one head
        {"attn_q.weight", 2, {E, KD}, kKda},
        {"attn_k.weight", 2, {E, KD}, kKda},
        {"attn_v.weight", 2, {E, VD}, kKda},
        {"ssm_conv1d_q.weight", 3, {g.ssm_d_conv, 1, KD}, kKda},
        {"ssm_conv1d_k.weight", 3, {g.ssm_d_conv, 1, KD}, kKda},
        {"ssm_conv1d_v.weight", 3, {g.ssm_d_conv, 1, VD}, kKda},
        {"ssm_f_a.weight", 2, {E, HD}, kKda},
        {"ssm_f_b.weight", 2, {HD, VD}, kKda},
        {"ssm_g_a.weight", 2, {E, HD}, kKda},
        {"ssm_g_b.weight", 2, {HD, VD}, kKda},
        {"ssm_beta.weight", 2, {E, g.ssm_v_heads}, kKda},
        {"ssm_a", 1, {g.ssm_v_heads}, kKda},
        {"ssm_dt.bias", 1, {VD}, kKda},
        {"ssm_norm.weight", 1, {HD}, kKda},
        {"attn_output.weight", 2, {VD, E}, kKda},
        // DSA: MLA without RoPE (q LoRA, one latent, k_b absorbed into q, v_b after attention) and the k-pool indexer
        {"attn_q_a.weight", 2, {E, g.q_lora}, kDsa},
        {"attn_q_a_norm.weight", 1, {g.q_lora}, kDsa},
        {"attn_q_b.weight", 2, {g.q_lora, g.n_head * g.head_dim}, kDsa},
        {"attn_kv_a_mqa.weight", 2, {E, g.kv_lora + g.n_rot}, kDsa},
        {"attn_kv_a_norm.weight", 1, {g.kv_lora}, kDsa},
        {"attn_k_b.weight", 3, {g.head_dim - g.n_rot, g.kv_lora, g.n_head}, kDsa},
        {"attn_v_b.weight", 3, {g.kv_lora, g.head_dim_v, g.n_head}, kDsa},
        {"attn_output.weight", 2, {g.n_head * g.head_dim_v, E}, kDsa},
        {"indexer.k_norm.weight", 1, {g.idx_key_dim}, kDsa},
        {"indexer.k_norm.bias", 1, {g.idx_key_dim}, kDsa},
        {"indexer.proj.weight", 2, {E, g.idx_q_heads}, kDsa},
        {"indexer.attn_k.weight", 2, {E, g.idx_key_dim}, kDsa},
        {"indexer.attn_q_b.weight", 2, {g.q_lora, g.idx_q_heads * g.idx_key_dim}, kDsa},
        {"indexer_compressor_gate.weight", 2, {E, g.idx_key_dim}, kDsa},
        {"indexer_compressor_ape.weight", 2, {g.idx_key_dim, g.idx_kpool}, kDsa},
        // the leading dense FFN
        {"ffn_gate.weight", 2, {E, g.n_ff_dense}, kDense},
        {"ffn_up.weight", 2, {E, g.n_ff_dense}, kDense},
        {"ffn_down.weight", 2, {g.n_ff_dense, E}, kDense},
        // MoE: sigmoid router with a selection bias, the shared expert
        {"ffn_gate_inp.weight", 2, {E, g.n_expert}, kMoe},
        {"exp_probs_b.bias", 1, {g.n_expert}, kMoe},
        {"ffn_gate_shexp.weight", 2, {E, shexp}, kMoe},
        {"ffn_up_shexp.weight", 2, {E, shexp}, kMoe},
        {"ffn_down_shexp.weight", 2, {shexp, E}, kMoe},
        {"ffn_gate_exps.weight", 3, {E, g.n_ff, g.n_expert}, kExperts},
        {"ffn_up_exps.weight", 3, {E, g.n_ff, g.n_expert}, kExperts},
        {"ffn_down_exps.weight", 3, {g.n_ff, E, g.n_expert}, kExperts},
        // the NextN block (llama.cpp loads it; no graph uses it yet)
        {"nextn.eh_proj.weight", 2, {2 * E, E}, kMtp},
        {"nextn.enorm.weight", 1, {E}, kMtp},
        {"nextn.hnorm.weight", 1, {E}, kMtp},
        {"nextn.shared_head_norm.weight", 1, {E}, kMtp, true},
    };
}

/// The tensor sets of layer `layer` (a trunk layer, or an MTP block when layer >= n_layers).
unsigned glm_layer_sets(const ModelGeometry& g, int64_t layer, bool with_experts) {
    const bool mtp = layer >= g.n_layers;
    const LayerKind k = layer_kind(g, layer);
    unsigned m = kNorm;
    if (!mtp) m |= kHc;
    else m |= kMtp;
    m |= (k == LayerKind::DsaMoe) ? kDsa : kKda;
    m |= (k == LayerKind::KdaDense) ? kDense : kMoe;
    if (with_experts && k != LayerKind::KdaDense) m |= kExperts;
    return m;
}

/// One glm5-next layer.  `full`: the lookup knows every dimension (a model file), so 3-D shapes and the experts
/// are compared; a pack's row is compared by ne0, ne1 (2-D) and the element count.
template <class Lookup>
bool check_one_glm(const Lookup& lk, const ModelGeometry& g, int64_t layer, bool full, std::string& err,
                   int64_t* checked) {
    const Names v{layer};
    const unsigned sets = glm_layer_sets(g, layer, full);
    const LayerKind kind = layer_kind(g, layer);
    for (const WantN& w : glm_wants(g)) {
        const Found r = lk(v.name(w.suffix));
        if (!(w.when & sets)) {
            // the other kind's signature tensors must be ABSENT: that is what ties the per-layer kinds (from
            // attention.head_count_kv) to the tensors, which no shape check can see
            const bool signature = (w.when == kKda && std::string(w.suffix) == "ssm_a") ||
                                   (w.when == kDsa && std::string(w.suffix) == "attn_kv_a_mqa.weight") ||
                                   (w.when == kDense && std::string(w.suffix) == "ffn_gate.weight") ||
                                   (w.when == kMoe && std::string(w.suffix) == "ffn_gate_inp.weight");
            if (signature && r.present) {
                err = "layer " + std::to_string(layer) + " is " + layer_kind_name(kind) + " by its geometry, but " +
                      v.name(w.suffix) + " is present";
                return false;
            }
            continue;
        }
        if (!r.present) {
            if (w.optional) continue;
            err = "layer " + std::to_string(layer) + " (" + layer_kind_name(kind) + "): missing " + v.name(w.suffix);
            return false;
        }
        int64_t want_elements = 1;
        for (int d = 0; d < w.nd; ++d) want_elements *= w.ne[d];
        if (w.nd == 1) {
            if (r.elements != want_elements) return fail(err, v, w.suffix, "elements", r.elements, want_elements);
        } else if (full) {
            static const char* dim[] = {"ne0", "ne1", "ne2"};
            if (r.nd != w.nd) return fail(err, v, w.suffix, "the dimension count", r.nd, w.nd);
            for (int d = 0; d < w.nd; ++d)
                if (r.ne[d] != w.ne[d]) return fail(err, v, w.suffix, dim[d], r.ne[d], w.ne[d]);
        } else {
            if (r.ne[0] != w.ne[0]) return fail(err, v, w.suffix, "ne0", r.ne[0], w.ne[0]);
            if (w.nd == 2 && r.ne[1] != w.ne[1]) return fail(err, v, w.suffix, "ne1", r.ne[1], w.ne[1]);
            if (r.elements != want_elements) return fail(err, v, w.suffix, "elements", r.elements, want_elements);
        }
        if (checked) ++*checked;
    }
    return true;
}

bool count_split(const ModelGeometry& g, int64_t n_qsa, int64_t n_gdn, std::string& err) {
    // The split is derived twice over in `docs/semantics.md` - the interval and an explicit ratio array -
    // and both give 36 GDN and 12 QSA.  Counting them here is the check that the LAYER TYPE PREDICATE and
    // the pack agree, which no per-tensor shape check can see.
    if (n_qsa != g.n_qsa_layers() || n_gdn != g.n_gdn_layers()) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "layer split is %lld QSA / %lld GDN, the geometry says %lld / %lld",
                      (long long) n_qsa, (long long) n_gdn, (long long) g.n_qsa_layers(),
                      (long long) g.n_gdn_layers());
        err = buf;
        return false;
    }
    return true;
}

}  // namespace

std::string LayerView::name(const char* suffix) const {
    return "blk." + std::to_string(layer_) + "." + suffix;
}

bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err) {
    if (layer < 0 || layer >= g.n_layers) {
        err = "layer " + std::to_string(layer) + " is outside 0.." + std::to_string(g.n_layers - 1);
        return false;
    }
    if (g.glm()) return check_one_glm(TableLookup{&table}, g, layer, false, err, nullptr);
    return check_one(TableLookup{&table}, g, layer, err);
}

bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err) {
    int64_t n_qsa = 0, n_gdn = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (g.glm() ? !check_one_glm(TableLookup{&table}, g, l, false, err, nullptr)
                    : !check_one(TableLookup{&table}, g, l, err))
            return false;
        if (is_qsa_layer(g, l)) {
            ++n_qsa;
        } else {
            ++n_gdn;
        }
    }
    return count_split(g, n_qsa, n_gdn, err);
}

bool check_all_gguf(const strata::GgufModel& model, const ModelGeometry& g, std::string& err, int64_t* checked) {
    const GgufLookup lk{&model};
    int64_t n = 0;
    // the vocabulary tensors, and the final norm (Qwen's output side is its gated residual, checked by its loader)
    struct Global {
        const char* name;
        int nd;
        int64_t ne0, ne1;
    };
    std::vector<Global> globals = {{"token_embd.weight", 2, g.n_embd, g.n_vocab}, {"output.weight", 2, g.n_embd, g.n_vocab}};
    if (g.glm()) globals.push_back({"output_norm.weight", 1, g.n_embd, 1});
    for (const Global& w : globals) {
        const Found r = lk(w.name);
        char buf[256];
        if (!r.present) { err = std::string("missing ") + w.name; return false; }
        if (r.nd != w.nd || r.ne[0] != w.ne0 || (w.nd == 2 && r.ne[1] != w.ne1)) {
            std::snprintf(buf, sizeof buf, "%s is [%lld, %lld], the kernels require [%lld, %lld]", w.name,
                          (long long) r.ne[0], (long long) r.ne[1], (long long) w.ne0, (long long) w.ne1);
            err = buf;
            return false;
        }
        ++n;
    }
    int64_t n_qsa = 0, n_gdn = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (g.glm() ? !check_one_glm(lk, g, l, true, err, &n) : !check_one(lk, g, l, err)) return false;
        if (is_qsa_layer(g, l)) ++n_qsa;
        else ++n_gdn;
    }
    for (int64_t l = g.n_layers; g.glm() && l < g.n_layers + g.n_layers_mtp; ++l)
        if (!check_one_glm(lk, g, l, true, err, &n)) return false;
    if (!count_split(g, n_qsa, n_gdn, err)) return false;
    if (checked) *checked = n;
    return true;
}

namespace {

/// A numeric metadata value as a signed integer (the reader keeps signed types sign-extended in `u`).
int64_t as_int(const strata::MetaValue& v) {
    return (v.type == strata::MetaType::F32 || v.type == strata::MetaType::F64) ? (int64_t) v.f : (int64_t) v.u;
}

/// A key that is one number, or a per-layer array whose `n` entries are all the same number (llama.cpp's
/// `get_key_or_arr`).  Strata's kernels take one value per model, so differing entries are refused by name.
bool uniform(const strata::GgufFile& m, const std::string& key, int64_t n, double& out, bool& present,
             std::string& err) {
    present = false;
    const strata::MetaValue* v = m.get(key);
    if (!v) return true;
    if (v->type == strata::MetaType::ARRAY) {
        if (v->count != (uint64_t) n || v->items.size() != v->count) {
            err = key + " has " + std::to_string(v->count) + " entries, the model has " + std::to_string(n) + " blocks";
            return false;
        }
        for (const auto& it : v->items) {
            if (!it.is_num()) { err = key + " holds a non-number"; return false; }
            if (it.num() != v->items[0].num()) {
                err = key + " differs between layers; this engine takes one value per model";
                return false;
            }
        }
        if (v->items.empty()) { err = key + " is empty"; return false; }
        out = v->items[0].num();
    } else {
        if (!v->is_num()) { err = key + " is not a number"; return false; }
        out = v->num();
    }
    present = true;
    return true;
}

std::string fmt_g(double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%g", v);
    return b;
}

}  // namespace

bool geometry_from_gguf(const strata::GgufFile& m, ModelGeometry& g, std::string& err) {
    const strata::MetaValue* arch = m.get("general.architecture");
    if (arch == nullptr || arch->s != "glm5-next") {
        // qwen4exp (and anything else, which its loaders refuse later, as before): exactly what generate.cpp read
        ModelGeometry q;
        if (const strata::MetaValue* v = m.get("qwen4exp.expert_count")) q.n_expert = (int64_t) v->u;
        if (const strata::MetaValue* v = m.get("qwen4exp.expert_used_count")) q.n_expert_used = (int64_t) v->u;
        g = q;
        return true;
    }
    err = strata::check_glm5_next(m);
    if (!err.empty()) {
        err = "glm5-next: " + err;
        return false;
    }
    auto i = [&](const char* key) { return as_int(*m.get(std::string("glm5-next.") + key)); };
    auto f = [&](const char* key) { return (float) m.get(std::string("glm5-next.") + key)->num(); };
    auto opt = [&](const char* key) { return m.get(std::string("glm5-next.") + key); };
    auto refuse = [&](const std::string& why) {
        err = "glm5-next: " + why;
        return false;
    };

    ModelGeometry r;
    r.arch = GeometryArch::Glm5Next;
    const int64_t n_all = i("block_count");
    const int64_t n_mtp = opt("nextn_predict_layers") ? i("nextn_predict_layers") : 0;
    if (n_all > kGeometryMaxLayers || n_mtp < 0 || n_mtp >= n_all)
        return refuse("block_count " + std::to_string(n_all) + " with " + std::to_string(n_mtp) +
                      " NextN blocks is outside what the geometry holds (" + std::to_string(kGeometryMaxLayers) + ")");
    r.n_layers = n_all - n_mtp;
    r.n_layers_mtp = n_mtp;
    r.n_embd = i("embedding_length");
    r.qsa_interval = 0;   // the layer types come from attention.head_count_kv, never from an interval

    // KDA: q, k and v each have `head_count` heads of `kda.head_dim`; the recurrent state is head_dim x head_dim per
    // head (llama.cpp: d_inner = head_dim * n_head for all three, G:89-91)
    const int64_t n_head = i("attention.head_count");
    r.ssm_state_size = i("kda.head_dim");
    r.ssm_k_heads = r.ssm_v_heads = n_head;
    r.ssm_d_conv = i("ssm.conv_kernel");
    r.ssm_conv_separate = true;
    r.ssm_qk_dim = r.ssm_state_size * n_head;
    r.ssm_value_dim = r.ssm_state_size * n_head;
    r.ssm_conv_channels = 2 * r.ssm_qk_dim + r.ssm_value_dim;   // Qwen's formula: q + k + v channels
    r.kda_gate_lower_bound = opt("kda.gate_lower_bound") ? f("kda.gate_lower_bound")
                                                          : -std::numeric_limits<float>::infinity();

    // DSA / MLA: 64 query heads over one latent row (the "kv head"), no RoPE
    r.n_head = n_head;
    r.n_head_kv = 1;
    r.head_dim = i("attention.key_length_mla");
    r.head_dim_v = i("attention.value_length_mla");
    r.q_lora = i("attention.q_lora_rank");
    r.kv_lora = i("attention.kv_lora_rank");
    r.n_rot = i("rope.dimension_count");
    r.idx_q_heads = i("attention.indexer.head_count");
    r.idx_key_dim = i("attention.indexer.key_length");
    r.idx_top_k = i("attention.indexer.top_k");
    r.idx_kpool = i("attention.indexer.kpool");
    r.idx_kpool_select_tail = opt("attention.indexer.kpool_select_tail") ? opt("attention.indexer.kpool_select_tail")->u != 0
                                                                         : true;   // llama.cpp's default
    if (!(r.idx_kpool > 1 && r.idx_top_k % r.idx_kpool == 0))   // llama.cpp asserts the same (G:43)
        return refuse("indexer top_k " + std::to_string(r.idx_top_k) + " is not a multiple of k-pool " +
                      std::to_string(r.idx_kpool) + " > 1");
    if (opt("attention.indexer.types"))
        return refuse("attention.indexer.types is set; this engine assumes every DSA layer has the full indexer");

    // mHC
    r.hc = i("hyper_connection.count");
    r.hc_lr = 0;
    r.hc_sinkhorn_iters = i("hyper_connection.sinkhorn_iterations");
    r.hc_eps = f("hyper_connection.epsilon");
    r.norm_rms_eps = f("attention.layer_norm_rms_epsilon");
    r.norm_eps = f("attention.layer_norm_epsilon");

    // FFN: the leading dense layers, then MoE
    double v = 0;
    bool present = false;
    r.n_expert = i("expert_count");
    r.n_expert_used = i("expert_used_count");
    r.n_expert_shared = i("expert_shared_count");
    if (!uniform(m, "glm5-next.expert_feed_forward_length", n_all, v, present, err)) return refuse(err);
    r.n_ff = (int64_t) v;
    if (!uniform(m, "glm5-next.feed_forward_length", n_all, v, present, err)) return refuse(err);
    r.n_ff_dense = (int64_t) v;
    r.n_dense_lead = i("leading_dense_block_count");
    if (r.n_expert < 1 || r.n_expert_used < 1 || r.n_expert_used > r.n_expert || r.n_ff < 1 || r.n_expert_shared < 0)
        return refuse("expert_count " + std::to_string(r.n_expert) + ", expert_used_count " +
                      std::to_string(r.n_expert_used) + ", expert_feed_forward_length " + std::to_string(r.n_ff) +
                      " do not make a mixture of experts");
    if (r.n_dense_lead < 0 || r.n_dense_lead >= r.n_layers || (r.n_dense_lead > 0 && r.n_ff_dense < 1))
        return refuse("leading_dense_block_count " + std::to_string(r.n_dense_lead) + " with feed_forward_length " +
                      std::to_string(r.n_ff_dense));
    const int64_t gating = opt("expert_gating_func") ? i("expert_gating_func") : 0;
    if (gating == 0 || gating == 2) r.gating = ExpertGating::Sigmoid;   // llama.cpp: none means sigmoid here (G:30-32)
    else if (gating == 1) r.gating = ExpertGating::Softmax;
    else return refuse("expert_gating_func " + std::to_string(gating) + " (only softmax 1 and sigmoid 2 are known)");
    r.expert_weights_scale = opt("expert_weights_scale") ? f("expert_weights_scale") : 0.0f;
    r.expert_weights_norm = opt("expert_weights_norm") ? opt("expert_weights_norm")->u != 0 : false;
    if (!uniform(m, "glm5-next.swiglu_clamp_exp", n_all, v, present, err)) return refuse(err);
    r.swiglu_clamp_exp = present ? (float) v : 0.0f;
    if (!uniform(m, "glm5-next.swiglu_clamp_shexp", n_all, v, present, err)) return refuse(err);
    r.swiglu_clamp_shexp = present ? (float) v : r.swiglu_clamp_exp;   // llama.cpp: shexp defaults to exp (G:36-38)

    // the vocabulary: the model's own key, else the tokenizer's token count
    if (opt("vocab_size")) r.n_vocab = i("vocab_size");
    else if (const strata::MetaValue* t = m.get("tokenizer.ggml.tokens"); t && t->type == strata::MetaType::ARRAY)
        r.n_vocab = (int64_t) t->count;
    else return refuse("neither glm5-next.vocab_size nor tokenizer.ggml.tokens gives the vocabulary size");

    // the per-layer kinds: llama.cpp's is_recr_impl[i] = n_head_kv(i) == 0 (G:21-23)
    const strata::MetaValue* kv = m.get("glm5-next.attention.head_count_kv");
    for (int64_t l = 0; l < n_all; ++l) {
        const strata::MetaValue& e = kv->items[(size_t) l];
        if (!e.is_num()) return refuse("attention.head_count_kv holds a non-number");
        const int64_t h = as_int(e);
        if (h < 0 || h > 1)
            return refuse("attention.head_count_kv[" + std::to_string(l) + "] is " + std::to_string(h) +
                          "; MLA layers have one latent (1), KDA layers none (0)");
        if (h == 1) {
            if (l < r.n_dense_lead) return refuse("layer " + std::to_string(l) + " is a dense DSA layer, which "
                                                  "this engine has no path for");
            r.attn_layer_mask |= 1ull << l;
        }
    }
    if (r.n_embd % 256 || r.ssm_qk_dim % 256 || r.n_ff % 256 || r.n_ff_dense % 256)
        return refuse("n_embd " + std::to_string(r.n_embd) + ", KDA width " + std::to_string(r.ssm_qk_dim) +
                      ", n_ff " + std::to_string(r.n_ff) + " and dense n_ff " + std::to_string(r.n_ff_dense) +
                      " must be whole 256-value blocks");
    g = r;
    return true;
}

std::string describe_geometry(const ModelGeometry& g, const char* p) {
    std::string s;
    char b[1024];
    if (!g.glm()) {
        std::snprintf(b, sizeof b, "%sqwen4exp geometry: n_embd %lld, %lld layers (%lld GDN + %lld QSA), %lld experts "
                                   "x %lld, top-%lld, vocabulary %lld\n",
                      p, (long long) g.n_embd, (long long) g.n_layers, (long long) g.n_gdn_layers(),
                      (long long) g.n_qsa_layers(), (long long) g.n_expert, (long long) g.n_ff,
                      (long long) g.n_expert_used, (long long) g.n_vocab);
        return b;
    }
    int64_t n_dense = 0, n_kda = 0, n_dsa = 0;
    std::string dsa_list, mtp;
    for (int64_t l = 0; l < g.n_layers + g.n_layers_mtp; ++l) {
        const LayerKind k = layer_kind(g, l);
        if (l >= g.n_layers) {
            mtp += (mtp.empty() ? "" : ", ") + std::to_string(l) + " " + layer_kind_name(k);
            continue;
        }
        if (k == LayerKind::KdaDense) ++n_dense;
        else if (k == LayerKind::KdaMoe) ++n_kda;
        else {
            ++n_dsa;
            dsa_list += (dsa_list.empty() ? "" : " ") + std::to_string(l);
        }
    }
    std::snprintf(b, sizeof b, "%sglm5-next geometry: n_embd %lld, %lld trunk layers + %lld NextN (MTP), vocabulary %lld\n",
                  p, (long long) g.n_embd, (long long) g.n_layers, (long long) g.n_layers_mtp, (long long) g.n_vocab);
    s += b;
    std::snprintf(b, sizeof b, "%slayers: %lld dense KDA (0-%lld), %lld KDA+MoE, %lld DSA+MoE (%s)%s%s\n", p,
                  (long long) n_dense, (long long) n_dense - 1, (long long) n_kda, (long long) n_dsa, dsa_list.c_str(),
                  mtp.empty() ? "" : "; MTP block ", mtp.c_str());
    s += b;
    std::snprintf(b, sizeof b, "%sKDA: %lld heads x %lld (state %lld x %lld per head), separate q/k/v conv1d of %lld "
                               "taps over %lld/%lld/%lld channels, gate lower bound %s\n",
                  p, (long long) g.ssm_v_heads, (long long) g.ssm_state_size, (long long) g.ssm_state_size,
                  (long long) g.ssm_state_size, (long long) g.ssm_d_conv, (long long) g.ssm_qk_dim,
                  (long long) g.ssm_qk_dim, (long long) g.ssm_value_dim, fmt_g(g.kda_gate_lower_bound).c_str());
    s += b;
    std::snprintf(b, sizeof b, "%sDSA/MLA: %lld query heads, q LoRA %lld, latent %lld (the cached row), key %lld / "
                               "value %lld per head, %s\n",
                  p, (long long) g.n_head, (long long) g.q_lora, (long long) g.kv_lora, (long long) g.head_dim,
                  (long long) g.head_dim_v,
                  g.n_rot == 0 ? "no RoPE" : ("RoPE " + std::to_string(g.n_rot) + " dims").c_str());
    s += b;
    std::snprintf(b, sizeof b, "%sDSA indexer: %lld heads x %lld, top-k %lld, k-pool %lld%s\n", p,
                  (long long) g.idx_q_heads, (long long) g.idx_key_dim, (long long) g.idx_top_k,
                  (long long) g.idx_kpool, g.idx_kpool_select_tail ? " (tail kept)" : "");
    s += b;
    std::snprintf(b, sizeof b, "%smHC: %lld streams (%lld wide), %lld mixing values per sublayer, Sinkhorn %lld "
                               "iterations, eps %s\n",
                  p, (long long) g.hc, (long long) g.hc_dim(), (long long) ((2 + g.hc) * g.hc),
                  (long long) g.hc_sinkhorn_iters, fmt_g(g.hc_eps).c_str());
    s += b;
    std::snprintf(b, sizeof b, "%sFFN: dense %lld wide; MoE %lld experts x %lld, top-%lld + %lld shared, %s gating, "
                               "weights %s, scale %s, SwiGLU clamp %s (experts) / %s (shared)\n",
                  p, (long long) g.n_ff_dense, (long long) g.n_expert, (long long) g.n_ff, (long long) g.n_expert_used,
                  (long long) g.n_expert_shared, g.gating == ExpertGating::Sigmoid ? "sigmoid" : "softmax",
                  g.expert_weights_norm ? "normalized" : "not normalized", fmt_g(g.expert_weights_scale).c_str(),
                  fmt_g(g.swiglu_clamp_exp).c_str(), fmt_g(g.swiglu_clamp_shexp).c_str());
    s += b;
    std::snprintf(b, sizeof b, "%snorms: RMS eps %s, LayerNorm eps %s\n", p, fmt_g(g.norm_rms_eps).c_str(),
                  fmt_g(g.norm_eps).c_str());
    s += b;
    return s;
}

}  // namespace strata::core
