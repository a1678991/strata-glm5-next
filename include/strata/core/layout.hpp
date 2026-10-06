// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>

namespace strata {
class GgufFile;
class GgufModel;
}

namespace strata::core {

/// The model family the geometry describes.  Every default below is Qwen3.8-Flash-Next's (`qwen4exp`); a `glm5-next`
/// geometry is filled from the model file by `geometry_from_gguf`.  The Qwen code paths read only the fields they
/// always read, with the values they always had.
enum class GeometryArch : int { Qwen4Exp = 0, Glm5Next = 1 };

/// What a layer is.  Qwen: GDN or QSA, both MoE.  GLM-5.3-Flash: llama.cpp's `is_recr` (attention.head_count_kv of
/// the layer == 0) is a KDA layer, otherwise a DSA (MLA + k-pool indexer) layer; the first `n_dense_lead` layers
/// have a dense FFN instead of experts.
enum class LayerKind : int { GdnMoe = 0, QsaMoe = 1, KdaDense = 2, KdaMoe = 3, DsaMoe = 4 };

/// glm5-next's router: llama.cpp's `expert_gating_func` (1 softmax, 2 sigmoid; 0 means sigmoid for this model).
enum class ExpertGating : int { Softmax = 1, Sigmoid = 2 };

/// The most layers a per-layer bit mask of the geometry holds (`attn_layer_mask`); the GGUF reader also keeps 64
/// items of a metadata array.
inline constexpr int64_t kGeometryMaxLayers = 64;

/// The model's geometry, taken from `docs/semantics.md` and the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
struct ModelGeometry {
    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // gated residual, on every layer
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;
    int64_t n_ff = 640;

    // ---- added for the second architecture.  The Qwen values are the ones the Qwen code already assumes (the
    // constants it has always used, or the keys generate.cpp has always read); no Qwen path reads a field below
    // that it did not read before, except `n_vocab` (the token embedding's row count, 248320 as before) and
    // `n_expert_used` (the top-k, `qwen4exp.expert_used_count`, 10 as before).
    GeometryArch arch = GeometryArch::Qwen4Exp;
    int64_t n_vocab = 248320;
    int64_t n_expert_used = 10;
    int64_t n_expert_shared = 1;

    // ---- glm5-next only (0 / false for qwen4exp unless noted)
    int64_t n_layers_mtp = 0;          ///< NextN blocks after the trunk (`nextn_predict_layers`), not in n_layers
    int64_t n_dense_lead = 0;          ///< leading layers with a dense FFN (`leading_dense_block_count`)
    int64_t n_ff_dense = 0;            ///< their FFN width (`feed_forward_length`)
    /// bit l set: layer l is attention (DSA), clear: recurrent (KDA).  From `attention.head_count_kv[l] > 0`, the
    /// rule of llama.cpp's `is_recr_impl` (src/models/glm5-next.cpp:21-23).  Bit n_layers.. are the MTP blocks.
    uint64_t attn_layer_mask = 0;
    bool ssm_conv_separate = false;    ///< q, k and v have their own conv1d (`ssm_conv1d_{q,k,v}`), not one fused one
    int64_t ssm_qk_dim = 0;            ///< channels of the q conv and of the k conv (= heads x head_dim)
    float kda_gate_lower_bound = 0;    ///< `kda.gate_lower_bound` (-5)
    int64_t q_lora = 0;                ///< MLA query LoRA rank (`attention.q_lora_rank`)
    int64_t kv_lora = 0;               ///< MLA latent width, the cached row (`attention.kv_lora_rank`)
    int64_t head_dim_v = 0;            ///< MLA value head width (`attention.value_length_mla`); head_dim is the key's
    int64_t n_rot = 0;                 ///< RoPE dims (`rope.dimension_count`); 0 = no RoPE
    int64_t idx_top_k = 0;             ///< indexer: cells kept (`attention.indexer.top_k`)
    int64_t idx_kpool = 0;             ///< indexer: keys pooled per row (`attention.indexer.kpool`)
    bool idx_kpool_select_tail = false;///< indexer: the pooled selection keeps the unpooled tail (llama.cpp default true)
    int64_t hc_sinkhorn_iters = 0;     ///< mHC (`hyper_connection.sinkhorn_iterations`)
    float hc_eps = 0;                  ///< mHC (`hyper_connection.epsilon`)
    float norm_rms_eps = 0;            ///< `attention.layer_norm_rms_epsilon`
    float norm_eps = 0;                ///< `attention.layer_norm_epsilon` (the indexer's LayerNorm)
    ExpertGating gating = ExpertGating::Softmax;
    float expert_weights_scale = 0;    ///< 0 = none
    bool expert_weights_norm = false;
    float swiglu_clamp_exp = 0;        ///< 0 = no clamp; one value for every layer (checked when read)
    float swiglu_clamp_shexp = 0;

    int64_t hc_dim() const { return hc * n_embd; }
    bool glm() const { return arch == GeometryArch::Glm5Next; }
    /// glm5-next: whether layer l (trunk or MTP) is a DSA layer.
    bool attn_layer(int64_t l) const { return l >= 0 && l < kGeometryMaxLayers && ((attn_layer_mask >> l) & 1u); }
    /// Qwen: `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    /// GLM: the DSA layers of the trunk, counted from the per-layer kinds.
    int64_t n_qsa_layers() const {
        if (!glm()) return n_layers / qsa_interval;
        int64_t n = 0;
        for (int64_t l = 0; l < n_layers; ++l) n += attn_layer(l) ? 1 : 0;
        return n;
    }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.  glm5-next: the
/// DSA layers, from the per-layer `attention.head_count_kv` (never from the interval).
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    if (g.glm()) return g.attn_layer(layer);
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// The kind of layer `layer` (0 <= layer < n_layers + n_layers_mtp).
inline LayerKind layer_kind(const ModelGeometry& g, int64_t layer) {
    if (!g.glm()) return is_qsa_layer(g, layer) ? LayerKind::QsaMoe : LayerKind::GdnMoe;
    if (g.attn_layer(layer)) return LayerKind::DsaMoe;   // geometry_from_gguf refuses a dense DSA layer
    return layer < g.n_dense_lead ? LayerKind::KdaDense : LayerKind::KdaMoe;
}

inline const char* layer_kind_name(LayerKind k) {
    switch (k) {
    case LayerKind::GdnMoe: return "GDN+MoE";
    case LayerKind::QsaMoe: return "QSA+MoE";
    case LayerKind::KdaDense: return "dense KDA";
    case LayerKind::KdaMoe: return "KDA+MoE";
    case LayerKind::DsaMoe: return "DSA+MoE";
    }
    return "?";
}

/// The geometry of a model from its metadata shard (`meta`, the first shard of a split model).  glm5-next: every
/// field from its keys, after `strata::check_glm5_next`, checked for the combinations the kernels assume.  Any
/// other file - qwen4exp - gets the defaults above and the two keys generate.cpp has always read here
/// (`qwen4exp.expert_count`, `qwen4exp.expert_used_count`), and nothing else: Qwen's architecture guard runs where it
/// always ran (the native loaders), so a Qwen file is refused, if at all, with the same message at the same point.
/// False with `err` set on a refusal.
bool geometry_from_gguf(const strata::GgufFile& meta, ModelGeometry& g, std::string& err);

/// One line per part of the geometry, in the engine's startup log style ("strata generate: ...").  GLM only today.
std::string describe_geometry(const ModelGeometry& g, const char* prefix);

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
/// glm5-next: the same over its tensor names (KDA, DSA, mHC, dense FFN, MoE), and a KDA layer carrying no MLA
/// tensor and a DSA layer no KDA tensor, so the per-layer kinds and the tensors agree.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

/// `check_all` against a model file's tensor directory instead of a pack: every tensor the layers need, at the
/// geometry's FULL shape (3-D included), plus the routed experts, the vocabulary tensors and the MTP block's
/// projection.  The engine runs it on a glm5-next file at startup; it reads headers only.  `checked` (optional)
/// receives the number of tensors whose shape was compared.
bool check_all_gguf(const strata::GgufModel& model, const ModelGeometry& g, std::string& err,
                    int64_t* checked = nullptr);

}  // namespace strata::core
