// src/core/glm_conversation.cpp - --serve's conversation cache for glm5-next: see include/strata/core/glm_conversation.hpp.
#include "strata/core/glm_conversation.hpp"

#include "strata/kernels/dsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace strata::core {
namespace sk = strata::kernels;

namespace {

bool fail(std::string& error, const std::string& message) {
    error = "glm conversation: " + message;
    return false;
}

bool cuda(cudaError_t e, const char* what, std::string& error) {
    if (e == cudaSuccess) return true;
    error = std::string("glm conversation: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool sync(const GlmSessionView& v, std::string& error) {
    return cuda(cudaStreamSynchronize((cudaStream_t) v.stream), "synchronize", error) &&
           cuda(cudaDeviceSynchronize(), "synchronize", error);
}

// device (or mapped host) bytes -> a buffer, and back; ConversationBuffer is segmented
bool to_buffer(ConversationBuffer& b, const void* src, size_t n, std::string& error) {
    b.resize(n);
    if (n == 0) return true;
    if (src == nullptr) return fail(error, "missing DSA cache array");
    bool ok = b.visit(0, n, [&](uint8_t* p, size_t m, size_t at) {
        return cudaMemcpy(p, (const uint8_t*) src + at, m, cudaMemcpyDefault) == cudaSuccess;
    });
    return ok || fail(error, "copying a DSA cache array out failed");
}

bool from_buffer(void* dst, const ConversationBuffer& b, std::string& error) {
    if (b.size() == 0) return true;
    if (dst == nullptr) return fail(error, "missing DSA cache array");
    const bool ok = b.visit(0, b.size(), [&](const uint8_t* p, size_t m, size_t at) {
        return cudaMemcpy((uint8_t*) dst + at, p, m, cudaMemcpyDefault) == cudaSuccess;
    });
    return ok || fail(error, "copying a DSA cache array in failed");
}

// the identity-layout arrays a layer's rows live in: the host copy of a streamed cache (authoritative there), else
// the VRAM slots (fully resident: slot == page)
struct Rows {
    void* lat = nullptr;
    void* scale = nullptr;
    void* kg = nullptr;
};
Rows rows_of(const sk::DsaKvCache& c) {
    const bool streamed = c.n_slots < c.n_pages;
    Rows r;
    if (c.fmt == sk::kDsaLatInt8) {
        r.lat = streamed ? (void*) c.host.lat_q : (void*) c.lat_q;
        r.scale = streamed ? (void*) c.host.lat_scale : (void*) c.lat_scale;
    } else {
        r.lat = streamed ? (void*) c.host.lat_f16 : (void*) c.lat_f16;
    }
    r.kg = streamed ? (void*) c.host.kg : (void*) c.kg;
    return r;
}

bool prefix_of(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

}  // namespace

std::array<int64_t, 18> glm_conversation_key(const ModelGeometry& g) {
    return {kGlmGeometryTag, g.n_embd, g.n_layers, g.n_dense_lead, (int64_t) g.attn_layer_mask, g.ssm_state_size,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_qk_dim, g.kv_lora, g.idx_key_dim, g.idx_kpool, g.idx_top_k, g.hc,
            g.n_expert, g.n_ff, g.n_head, g.n_vocab};
}

bool glm_conversation_sizes(const GlmSessionView& v, GlmConversationSizes& z, std::string& error) {
    z = {};
    if (v.g == nullptr || v.cfg == nullptr || !v.g->glm()) return fail(error, "not a glm5-next session");
    const ModelGeometry& g = *v.g;
    if (v.kda_hist.size() != (size_t) g.n_layers || v.kda_state.size() != (size_t) g.n_layers ||
        v.dsa.size() != (size_t) g.n_layers)
        return fail(error, "the session view does not cover every layer");
    if (g.ssm_qk_dim <= 0 || g.ssm_d_conv < 2 || g.ssm_v_heads <= 0 || g.ssm_state_size <= 0)
        return fail(error, "invalid KDA geometry");
    z.kda_hist = (size_t) 3 * (size_t) g.ssm_qk_dim * (size_t) (g.ssm_d_conv - 1) * sizeof(float);
    z.kda_state = (size_t) g.ssm_v_heads * (size_t) g.ssm_state_size * (size_t) g.ssm_state_size * sizeof(float);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (g.attn_layer(l)) {
            ++z.n_dsa;
            if (v.dsa[(size_t) l].page_table == nullptr) return fail(error, "a DSA layer without its cache");
        } else {
            ++z.n_kda;
            if (v.kda_hist[(size_t) l] == nullptr || v.kda_state[(size_t) l] == nullptr)
                return fail(error, "a KDA layer without its state");
        }
    }
    z.running = (size_t) z.n_kda * (z.kda_hist + z.kda_state);
    const sk::DsaShapes& s = v.dshape;
    if (z.n_dsa > 0) {
        if (s.kv_lora <= 0 || s.idx_dim <= 0 || s.kpool <= 0) return fail(error, "invalid DSA geometry");
        const int fmt = v.cfg->kv_fmt;
        z.lat_row = fmt == sk::kDsaLatInt8 ? (size_t) s.kv_lora : (size_t) s.kv_lora * 2;
        z.scale_row = fmt == sk::kDsaLatInt8 ? (size_t) (s.kv_lora / sk::DSA_LAT_GROUP) * 2 : 0;
        z.kg_row = (size_t) 2 * (size_t) s.idx_dim * 2;
        z.pooled_row = (size_t) s.idx_dim * 2;
        z.kpool = s.kpool;
    }
    return true;
}

bool glm_checkpoint_validate(const ConversationCheckpoint& c, const GlmSessionView& v, std::string& error) {
    GlmConversationSizes z;
    if (!glm_conversation_sizes(v, z, error)) return false;
    if (c.ids.empty() || (int64_t) c.ids.size() > v.cfg->max_cells || !c.stage_parts.empty() || !c.imgs.empty())
        return fail(error, "invalid checkpoint tokens");
    if (std::any_of(c.ids.begin(), c.ids.end(), [&](int32_t id) { return id < 0 || id >= v.g->n_vocab; }))
        return fail(error, "a checkpoint token outside the vocabulary");
    if (c.gdn.size() != z.running || !c.ple.empty() || !c.tails.empty() || !c.dead.empty() || !c.block_pos.empty())
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

bool glm_checkpoint_save(ConversationCheckpoint& c, const GlmSessionView& v, std::string& error) {
    GlmConversationSizes z;
    if (!glm_conversation_sizes(v, z, error) || !sync(v, error)) return false;
    c.gdn.resize(z.running);
    c.ple.clear(); c.tails.clear(); c.dead.clear(); c.block_pos.clear();
    size_t at = 0;
    for (int64_t l = 0; l < v.g->n_layers; ++l) {
        if (v.g->attn_layer(l)) continue;
        if (!cuda(cudaMemcpy(c.gdn.data() + at, v.kda_hist[(size_t) l], z.kda_hist, cudaMemcpyDeviceToHost),
                  "KDA history out", error) ||
            !cuda(cudaMemcpy(c.gdn.data() + at + z.kda_hist, v.kda_state[(size_t) l], z.kda_state,
                             cudaMemcpyDeviceToHost), "KDA state out", error))
            return false;
        at += z.kda_hist + z.kda_state;
    }
    return true;
}

bool glm_checkpoint_restore(const ConversationCheckpoint& c, const GlmSessionView& v, std::string& error) {
    if (!glm_checkpoint_validate(c, v, error) || !sync(v, error)) return false;
    GlmConversationSizes z;
    if (!glm_conversation_sizes(v, z, error)) return false;
    size_t at = 0;
    for (int64_t l = 0; l < v.g->n_layers; ++l) {
        if (v.g->attn_layer(l)) continue;
        if (!cuda(cudaMemcpy(v.kda_hist[(size_t) l], c.gdn.data() + at, z.kda_hist, cudaMemcpyHostToDevice),
                  "KDA history in", error) ||
            !cuda(cudaMemcpy(v.kda_state[(size_t) l], c.gdn.data() + at + z.kda_hist, z.kda_state,
                             cudaMemcpyHostToDevice), "KDA state in", error))
            return false;
        at += z.kda_hist + z.kda_state;
    }
    return sync(v, error);
}

bool glm_snapshot_bytes(const ConversationView& view, const GlmSessionView& v, size_t& bytes, std::string& error) {
    bytes = 0;
    GlmConversationSizes z;
    if (!glm_conversation_sizes(v, z, error)) return false;
    const size_t L = view.ids.size();
    if (L == 0 || (int64_t) L > v.cfg->max_cells) return fail(error, "invalid live tokens");
    bytes += L * sizeof(int32_t) + z.running;
    bytes += view.checkpoints.size() * sizeof(ConversationCheckpoint);
    for (const auto& c : view.checkpoints) bytes += c.ids.size() * sizeof(int32_t) + c.gdn.size();
    bytes += (size_t) z.n_dsa * sizeof(ConversationKv);
    if (z.n_dsa > 0)
        bytes += (size_t) z.n_dsa * (L * (z.lat_row + z.scale_row + z.kg_row) + (L / (size_t) z.kpool) * z.pooled_row);
    // the buffers' segment directories (one 24-byte entry per 16 MiB segment of each of 4 buffers) and the vectors'
    // slack: an over-estimate by design (make_room then evicts a little early; never too late)
    bytes += (size_t) z.n_dsa * (4 * sizeof(std::vector<uint8_t>) * (L * 1024 / ConversationBuffer::segment_bytes + 2)) +
             4096;
    return true;
}

bool glm_snapshot_save(SavedConversation& image, const ConversationView& view, const GlmSessionView& v,
                       std::string& error) {
    GlmConversationSizes z;
    if (!glm_conversation_sizes(v, z, error)) return false;
    const int64_t L = (int64_t) view.ids.size();
    if (L == 0 || L > v.cfg->max_cells || !view.images.empty()) return fail(error, "invalid live tokens");
    for (const auto& c : view.checkpoints) {
        if (!prefix_of(c.ids, view.ids)) return fail(error, "checkpoint is not a live token prefix");
        if (!glm_checkpoint_validate(c, v, error)) return false;
    }
    if (!sync(v, error)) return false;
    SavedConversation out;   // built aside: a failure publishes nothing
    out.geometry = glm_conversation_key(*v.g);
    out.layer_lo = 0;
    out.layer_hi = v.g->n_layers;
    out.cvec = view.cvec;
    out.live.ids = view.ids;
    if (!glm_checkpoint_save(out.live, v, error)) return false;
    out.checkpoints = view.checkpoints;
    const sk::DsaShapes& s = v.dshape;
    out.kv.reserve((size_t) z.n_dsa);
    for (int64_t l = 0; l < v.g->n_layers; ++l) {
        if (!v.g->attn_layer(l)) continue;
        const sk::DsaKvCache& c = v.dsa[(size_t) l];
        const Rows r = rows_of(c);
        ConversationKv kv;
        kv.format = kGlmKvFormat | c.fmt;
        kv.cells = L;
        kv.heads = 1;
        kv.head_dim = s.kv_lora;
        kv.page_size = s.kpool;
        kv.pooled_rows = L / s.kpool;
        kv.idx_dim = s.idx_dim;
        if (!to_buffer(kv.k, r.lat, (size_t) L * z.lat_row, error) ||
            !to_buffer(kv.k_scale, r.scale, (size_t) L * z.scale_row, error) ||
            !to_buffer(kv.v, r.kg, (size_t) L * z.kg_row, error) ||
            !to_buffer(kv.pooled, c.pooled, (size_t) kv.pooled_rows * z.pooled_row, error))
            return false;
        out.kv.push_back(std::move(kv));
    }
    image = std::move(out);
    return true;
}

bool glm_snapshot_validate(const SavedConversation& image, const GlmSessionView& v, std::string& error) {
    GlmConversationSizes z;
    if (!glm_conversation_sizes(v, z, error)) return false;
    if (image.geometry != glm_conversation_key(*v.g)) return fail(error, "incompatible runtime geometry");
    if (image.layer_lo != 0 || image.layer_hi != v.g->n_layers || !image.stage_images.empty())
        return fail(error, "an image of another layer range");
    if (!glm_checkpoint_validate(image.live, v, error)) return false;
    const int64_t L = (int64_t) image.live.ids.size();
    for (const auto& c : image.checkpoints) {
        if (!prefix_of(c.ids, image.live.ids)) return fail(error, "checkpoint is not a live token prefix");
        if (!glm_checkpoint_validate(c, v, error)) return false;
    }
    if ((int64_t) image.kv.size() != z.n_dsa) return fail(error, "invalid DSA layer count");
    const sk::DsaShapes& s = v.dshape;
    for (const auto& kv : image.kv) {
        if (kv.format != (kGlmKvFormat | v.cfg->kv_fmt) || kv.cells != L || kv.heads != 1 || kv.head_dim != s.kv_lora ||
            kv.page_size != s.kpool || kv.pooled_rows != L / s.kpool || kv.idx_dim != s.idx_dim)
            return fail(error, "a DSA layer of another format or length");
        if (kv.k.size() != (size_t) L * z.lat_row || kv.k_scale.size() != (size_t) L * z.scale_row ||
            kv.v.size() != (size_t) L * z.kg_row || !kv.v_scale.empty() ||
            kv.pooled.size() != (size_t) kv.pooled_rows * z.pooled_row)
            return fail(error, "a DSA layer's payload has the wrong size");
    }
    return true;
}

ConversationRestore glm_snapshot_restore(const SavedConversation& image, const GlmSessionView& v, std::string& error) {
    if (!glm_snapshot_validate(image, v, error)) return ConversationRestore::invalid;
    if (!sync(v, error)) return ConversationRestore::transfer_failed;
    size_t k = 0;
    for (int64_t l = 0; l < v.g->n_layers; ++l) {
        if (!v.g->attn_layer(l)) continue;
        sk::DsaKvCache c = v.dsa[(size_t) l];
        // every page evicted (streamed) or cleared, the pooled rows zero: the cells past the image are unwritten
        sk::dsa_kv_reset(c, v.dshape, v.stream);
        if (!sync(v, error)) return ConversationRestore::transfer_failed;
        const Rows r = rows_of(c);
        const ConversationKv& kv = image.kv[k++];
        if (!from_buffer(r.lat, kv.k, error) || !from_buffer(r.scale, kv.k_scale, error) ||
            !from_buffer(r.kg, kv.v, error) || !from_buffer(c.pooled, kv.pooled, error))
            return ConversationRestore::transfer_failed;
    }
    if (!glm_checkpoint_restore(image.live, v, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}

}  // namespace strata::core
