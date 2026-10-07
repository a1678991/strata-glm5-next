// include/strata/core/glm.hpp - GLM-5.3-Flash (llama.cpp arch `glm5-next`) decode: the layers composed, the per-
// sequence state, and the verify window with its commit.  The fork's second model, beside the Qwen path, which this
// file does not touch: everything here is reached only for a geometry with `arch == Glm5Next`.
//
// THE REFERENCE is llama.cpp f1cee99, src/models/glm5-next.cpp ("G"), the graph constructor (G:557-680):
//
//   x[t][h]   = token_embd[tok_t]                        for the 4 hyper-connection streams h (hc_init: a repeat)
//   per layer l:
//     cur     = hc_pre(x, hc_attn_*)                     mhc_flat_norm, the hc_attn_fn GEMV, mhc_pre
//     cur     = rms_norm(cur) * attn_norm
//     f       = KDA(cur) if head_count_kv[l] == 0, else DSA(cur)           (build_kda_layer / build_dsa_layer)
//     x       = hc_post(f, x, post, comb)                mhc_post
//     cur     = hc_pre(x, hc_ffn_*);  cur = rms_norm(cur) * ffn_norm
//     f       = clamped SwiGLU FFN (l < leading_dense_block_count), else MoE(cur) + shared expert(cur)
//     x       = hc_post(f, x, post, comb)                -> llama.cpp's node "l_out-<l>", F32 [n_embd, hc, T]
//   head      = rms_norm(mean_h x[h]) * output_norm;  logits = output . head
//
// Each piece is a wave-1 kernel with its own parity test against ggml-cpu at f1cee99 (kda_decode.hpp, mhc.hpp,
// ffn_clamp.hpp, router_sigmoid.hpp, dsa.hpp); this file owns the ORDER, the buffers and the state.
//
// THE WINDOW.  `run` takes T <= max_t tokens at positions pos0 .. pos0+T-1 through the layers and the head, and
// leaves the KDA state alone: the KDA kernels run in their verify mode (history and state read only), with each
// layer's raw q|k|v, conv output, f_b output and beta logits kept per layer.  `commit(n_keep)` then replays the first
// n_keep tokens into every KDA layer's conv history and state (kda_conv_commit / kda_step_commit, bitwise the state of
// n_keep single steps: kda_parity).  The DSA latent cache is appended for all T positions, as the QSA KV is: a cell
// past the kept prefix is written again when its position is processed again, before any query reads it (a pool is
// re-pooled when its last cell is appended again).  T = 1 with commit(1) is plain decode.
//
// THE DUMP.  `set_dump(layer, writer)` copies x after layer `layer` ([T][hc][n_embd] F32, glm-splice's l_out row
// layout: stream s, element i at s * n_embd + i) to a GLMT file per layer, one row per token, so glm-splice can
// replace llama.cpp's own `l_out-<layer>` with ours (validate/README.md).
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/dsa.hpp"
#include "strata/kernels/ffn_clamp.hpp"
#include "strata/kernels/kda_decode.hpp"
#include "strata/kernels/mhc.hpp"
#include "strata/kernels/sampler.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class NativeEmbed;
class NativeHead;
class GlmExperts;

/// One layer's weights, resolved by name from the pack's table (F32 rows in the arena) and NativeDense (every
/// quantized non-expert tensor in its GGUF form, 3-D ones flattened).  Device pointers; `*_type` are ggml type ids.
struct GlmLayerWeights {
    int64_t layer = 0;
    LayerKind kind = LayerKind::KdaDense;
    // mHC, both halves: hc_*_fn [hc * n_embd -> (2 + hc) * hc] quantized, scale [3], base [(2 + hc) * hc] F32
    const void* hc_attn_fn = nullptr;  int hc_attn_fn_type = -1;
    const float* hc_attn_scale = nullptr;
    const float* hc_attn_base = nullptr;
    const void* hc_ffn_fn = nullptr;   int hc_ffn_fn_type = -1;
    const float* hc_ffn_scale = nullptr;
    const float* hc_ffn_base = nullptr;
    const float* attn_norm = nullptr;  ///< [n_embd]
    const float* ffn_norm = nullptr;   ///< [n_embd]
    // KDA (head_count_kv[l] == 0)
    const void* wq = nullptr; int wq_type = -1;   ///< attn_q [n_embd -> d_inner]
    const void* wk = nullptr; int wk_type = -1;   ///< attn_k
    const void* wv = nullptr; int wv_type = -1;   ///< attn_v
    const float* conv_q = nullptr;                ///< ssm_conv1d_q [d_inner][d_conv]
    const float* conv_k = nullptr;
    const float* conv_v = nullptr;
    const void* f_a = nullptr; int f_a_type = -1; ///< ssm_f_a [n_embd -> head_dim]
    const void* f_b = nullptr; int f_b_type = -1; ///< ssm_f_b [head_dim -> d_inner]
    const void* g_a = nullptr; int g_a_type = -1; ///< ssm_g_a [n_embd -> head_dim]
    const void* g_b = nullptr; int g_b_type = -1; ///< ssm_g_b [head_dim -> d_inner]
    const void* beta = nullptr; int beta_type = -1; ///< ssm_beta [n_embd -> n_head]
    const float* dt_b = nullptr;                  ///< ssm_dt.bias [d_inner]
    const float* ssm_a = nullptr;                 ///< ssm_a [n_head] (= -exp(A_log), as stored)
    const float* ssm_norm = nullptr;              ///< ssm_norm [head_dim]
    const void* wo = nullptr; int wo_type = -1;   ///< attn_output [d_inner -> n_embd]
    // DSA (head_count_kv[l] == 1)
    strata::kernels::DsaWeights dsa;
    // FFN: dense (l < n_dense_lead) or the shared expert of an MoE layer
    strata::kernels::FfnClampWeights ffn;         ///< ffn_{gate,up,down} or ffn_*_shexp
    int64_t ffn_n_ff = 0;                         ///< its width (the gate's rows)
    // MoE router
    const float* router = nullptr;                ///< ffn_gate_inp F32 [n_embd -> n_expert]
    const float* exp_probs_b = nullptr;           ///< [n_expert]
};

/// Resolves every layer's weights.  False with `err` naming the first missing or unserved tensor (a quantized tensor
/// must have been attached by NativeDense: GLM has no canonical form for them).
bool glm_bind_weights(const WeightTable& wt, const ModelGeometry& g, std::vector<GlmLayerWeights>& out,
                      std::string& err);

/// A GLMT file (validate/scripts/glmt.py, glm-splice --replace): a 128-byte header patched at `finish`, then one row
/// of `row_elems` F32 per token.
class GlmtWriter {
public:
    GlmtWriter() = default;
    ~GlmtWriter() { finish(); }
    GlmtWriter(const GlmtWriter&) = delete;
    GlmtWriter& operator=(const GlmtWriter&) = delete;
    bool create(const std::string& path, const std::string& name, int64_t ne0, int64_t ne1, std::string& err);
    bool append(const float* rows, int64_t n_rows);
    bool finish();
    int64_t rows() const { return rows_; }

private:
    std::FILE* f_ = nullptr;
    std::string name_;
    int64_t ne0_ = 0, ne1_ = 0, rows_ = 0;
};

/// The run's settings.
struct GlmConfig {
    int max_t = 8;              ///< the widest window (1..8)
    int64_t max_cells = 4096;   ///< context capacity (the DSA caches)
    int64_t n_run = -1;         ///< run layers [0, n_run) only (-1: all); the head runs only when all do
    int kv_fmt = strata::kernels::kDsaLatInt8;   ///< the DSA latent format (int8 production, f16 parity)
    /// --kv-resident N (wave 6, serving): the DSA latent cache keeps N cells per layer in VRAM and every cell in
    /// page-locked, device-mapped host memory (dsa.hpp "STREAMING"); 0, or N >= max_cells: fully resident.  The decode
    /// windows make the pages they select resident (dsa_kv_stream_resolve), the prompt path reads a page that is not
    /// resident from the host copy.  The int8 latent only.
    int64_t kv_resident = 0;
};

/// The DSA layers' VRAM pages for `cfg` (0: fully resident, every page in VRAM).
int64_t glm_kv_slots(const ModelGeometry& g, const GlmConfig& cfg);
/// The smallest --kv-resident (cells) a window of `max_t` tokens can stream with: every token's selection (512 pools
/// and its tail page) resident at once, plus one page, and never fewer than kv_stream.cu's 1,024-page resolve block.
int64_t glm_kv_resident_min(const ModelGeometry& g, int max_t);
/// Page-locked host bytes of the streamed DSA caches (0 when fully resident).
uint64_t glm_session_host_bytes(const ModelGeometry& g, const GlmConfig& cfg);

/// Bytes of the GLM session's device arena (state + window buffers) for `cfg`: every KDA layer's conv history and
/// recurrent state and its window record, every DSA layer's latent cache, and the shared per-window scratch.
uint64_t glm_session_bytes(const ModelGeometry& g, const GlmConfig& cfg);

/// The prompt path's view of a session (src/prefill/glm_prefill.cpp, port plan Stage 6): the weights, the shapes and
/// the state it reads and writes in place - every KDA layer's conv history and recurrent state, every DSA layer's
/// latent cache - so the decode windows continue exactly where the prompt left them.  Device pointers.
struct GlmSessionView {
    const ModelGeometry* g = nullptr;
    const GlmConfig* cfg = nullptr;
    const std::vector<GlmLayerWeights>* w = nullptr;
    const NativeEmbed* embed = nullptr;
    strata::kernels::KdaShape kshape;
    strata::kernels::MhcShape mshape;
    strata::kernels::DsaShapes dshape;
    std::vector<float*> kda_hist;                     ///< per layer: [3 * d_inner][d_conv - 1], null for DSA layers
    std::vector<float*> kda_state;                    ///< per layer: [n_head][head_dim][head_dim]
    std::vector<strata::kernels::DsaKvCache> dsa;     ///< per layer (unused rows for KDA layers)
    void* stream = nullptr;                           ///< the session's stream
    int64_t n_run = 0;
};

class GlmEngine {
public:
    GlmEngine() = default;
    ~GlmEngine();
    GlmEngine(const GlmEngine&) = delete;
    GlmEngine& operator=(const GlmEngine&) = delete;

    /// Binds the weights, allocates and zeroes the session.  `head` may be null (then n_run must stop before the
    /// last layer or the logits are not made); `experts` may be null while no MoE layer runs.
    bool init(const WeightTable& wt, const ModelGeometry& g, const NativeEmbed& embed, const NativeHead* head,
              GlmExperts* experts, const GlmConfig& cfg, std::string& err);
    /// A fresh sequence: every KDA state and history to zero, every DSA cache emptied.
    bool reset(std::string& err);

    /// One window: tokens[0..T) at positions pos0.. through layers [0, n_run) and, when every layer ran, the head.
    /// Synchronizes before returning.  The KDA state is not advanced until `commit`.
    /// `x_in` (host, [T][hc][n_embd], optional): the streams entering layer `from` instead of the embedding - the
    /// rows of llama.cpp's l_out-<from - 1>, so that layers [from, n_run) are checked on the reference's input alone.
    bool run(int T, const int32_t* tokens, int64_t pos0, std::string& err, const float* x_in = nullptr,
             int64_t from = 0);
    /// Keep the first n_keep (1..T) tokens of the last window.
    bool commit(int n_keep, std::string& err);

    /// Row t of the last window's logits (n_vocab floats) to the host.
    bool copy_logits(int t, float* host) const;
    /// The last window's greedy picks (argmax of each row).
    bool argmax(int T, int32_t* out) const;
    /// The last window's (or head_rows') picks with the sampler chain (kernels/sampler.hpp, the Qwen verifier's head
    /// sampling): row t draws Philox(p.seed, pos0 + t), so a seed replays the same text.  `history` (host, [T][h], may be
    /// null with h = 0): each row's penalty window, sampler.hpp's penalty_rows.  h is at most kSampleHistoryCap.
    static constexpr int kSampleHistoryCap = 4096;
    bool sample(int T, const strata::kernels::SamplerParams& p, const int32_t* history, int h, int32_t* out,
                std::string& err);
    /// STRATA_LOGPOS's rows (Verifier::window_logprobs' format and arithmetic) for the last window.
    bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                         std::string& err) const;

    /// Dump x after `layer` to `w` for every window from now on (null stops it).
    void set_dump(int64_t layer, GlmtWriter* w);
    /// Debugging: the intermediate nodes of `layer` - llama.cpp's names, one GLMT file each in `dir`: hc_mixes#0/#1,
    /// hc_attn_pre, attn_norm, kda_out or attn_out, hc_attn_post, hc_ffn_pre, ffn_norm, ffn_out (STRATA_GLM_NODES).
    bool set_node_dump(int64_t layer, const std::string& dir, std::string& err);

    /// The prompt path: the session's state and weights (see GlmSessionView).
    GlmSessionView session_view() const;
    /// The head over `n` (1..max_t) rows of hyper-connection streams at `x_rows` (device, [n][hc][n_embd]): the logits
    /// of the prompt's last positions pos0 .. pos0 + n - 1, read by copy_logits / argmax / window_logprobs like a
    /// window's rows.  The KDA state is not touched, and there is no window to commit afterwards.
    bool head_rows(const float* x_rows, int n, int64_t pos0, std::string& err);
    /// Debugging and the Stage 6 check: the session state after the first `n_cells` positions, one GLMT file per
    /// tensor in `dir` - kda_hist-<l> (3 x d_inner rows of d_conv - 1), kda_state-<l> (n_head x head_dim rows of
    /// head_dim), dsa_latent-<l> (n_cells rows of kv_lora, dequantized as the attention reads it), dsa_kg-<l>
    /// (n_cells rows of key | gate), dsa_pooled-<l> (n_cells / kpool rows of idx_dim).  Layers [0, n_run).
    bool dump_state(const std::string& dir, int64_t n_cells, std::string& err) const;

    bool head_runs() const { return head_ != nullptr && n_run_ == g_.n_layers; }
    int64_t n_run() const { return n_run_; }
    int64_t vocab() const { return g_.n_vocab; }
    void* stream() const { return stream_; }
    uint64_t session_bytes() const { return arena_bytes_; }
    /// The page-locked host copy of the streamed DSA caches (0: fully resident).
    uint64_t session_host_bytes() const { return host_kv_bytes_; }
    const GlmConfig& config() const { return cfg_; }

    /// Per-layer GPU times of the windows so far (STRATA_GLM_PROFILE=1), one line; empty when off.
    std::string profile_report() const;

private:
    struct KdaLayer {
        float* hist = nullptr;    ///< [3 * d_inner][d_conv - 1]
        float* state = nullptr;   ///< [n_head][head_dim][head_dim]
        float* qkv = nullptr;     ///< window: [max_t][3 * d_inner] raw projections
        float* h = nullptr;       ///< window: [max_t][3 * d_inner] conv + SiLU (+ L2)
        float* fb = nullptr;      ///< window: [max_t][d_inner] f_b(f_a x), without dt_b
        float* braw = nullptr;    ///< window: [max_t][n_head]
    };
    bool layer(int64_t l, int T, std::string& err);
    bool mhc_half(const GlmLayerWeights& w, bool ffn, int T);
    bool kda(int64_t l, const GlmLayerWeights& w, int T);
    bool head(int T, std::string& err);

    ModelGeometry g_;
    GlmConfig cfg_;
    std::vector<GlmLayerWeights> w_;
    const NativeEmbed* embed_ = nullptr;
    const NativeHead* head_ = nullptr;
    GlmExperts* experts_ = nullptr;
    const float* output_norm_ = nullptr;
    int64_t n_run_ = 0;
    void* stream_ = nullptr;
    void* arena_ = nullptr;
    uint64_t arena_bytes_ = 0;
    void* host_kv_ = nullptr;          ///< --kv-resident: the DSA caches' host copies (page-locked, mapped)
    uint64_t host_kv_bytes_ = 0;
    int32_t* d_sample_ = nullptr;      ///< sample(): [max_t] picks, then [max_t][kSampleHistoryCap] histories
    strata::kernels::KdaShape kshape_;
    strata::kernels::MhcShape mshape_;
    strata::kernels::DsaShapes dshape_;
    std::vector<KdaLayer> kda_;                          ///< per layer (unused rows for DSA layers)
    std::vector<strata::kernels::DsaKvCache> dsa_;       ///< per layer (unused rows for KDA layers)
    strata::kernels::DsaScratch dsc_;
    // window buffers (device)
    int32_t* tok_ = nullptr;      ///< [max_t]
    int32_t* steps_ = nullptr;    ///< [max_t][kStepCount]
    int32_t* nkeep_ = nullptr;    ///< [1]
    int32_t* h_nkeep_ = nullptr;  ///< pinned host staging of nkeep_
    int32_t* h_steps_ = nullptr;  ///< pinned host staging of steps_
    int32_t* h_tok_ = nullptr;    ///< pinned host staging of tok_
    float *emb_ = nullptr, *x_ = nullptr, *flat_ = nullptr, *mixes_ = nullptr, *post_ = nullptr, *comb_ = nullptr;
    float *cur_ = nullptr, *xn_ = nullptr, *f_ = nullptr;
    void* xq_ = nullptr;          ///< Q8_1 of the widest activation (hc * n_embd) x max_t
    void* sq_ = nullptr;          ///< Q8_1 of a d_inner-wide activation x max_t
    float *tmp3_ = nullptr, *fa_ = nullptr, *ga_ = nullptr, *gb_ = nullptr, *y_ = nullptr;
    void* ffn_scratch_ = nullptr;
    float *hm_ = nullptr, *hn_ = nullptr, *logits_ = nullptr;
    float* h_dump_ = nullptr;     ///< pinned host [max_t][hc][n_embd]
    std::vector<GlmtWriter*> dump_;
    int64_t node_layer_ = -1;
    std::vector<std::unique_ptr<GlmtWriter>> nodes_;   ///< in the order of node_names()
    std::vector<float> h_node_;
    bool dump_node(int i, const float* dev, int64_t per_token, int T, std::string& err);
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    // STRATA_GLM_PROFILE
    bool prof_ = false;
    std::vector<void*> ev_;                              ///< per layer and the head: start/stop events
    std::vector<double> prof_ms_;
    int64_t prof_windows_ = 0;
};

}  // namespace strata::core
