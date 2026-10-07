// include/strata/core/glm_conversation.hpp - --serve's conversation cache for GLM-5.3-Flash (glm5-next): what a
// GlmEngine session leaves behind, in the containers the Qwen path parks (conversation_cache.hpp), so the RAM tier
// (ConversationCache) carries GLM conversations unchanged.
//
// The session splits as the Qwen one does (generate.cpp, "--serve's conversation cache"):
//   * RUNNING state - every KDA layer's conv history ([3 * d_inner][d_conv - 1] F32) and recurrent state
//     ([n_head][head_dim][head_dim] F32): 4,489,216 B per layer, 34 layers = 152.6 MB on GLM-5.3-Flash.  It describes
//     "everything so far" and cannot be rewound, so a checkpoint is a copy of exactly it (ConversationCheckpoint::gdn,
//     the layers in order, each layer's history then its state; the Qwen-only fields stay empty).
//   * POSITIONAL state - every DSA layer's latent cache (int8 codes + f16 scales, or the f16 parity latent), the
//     indexer's key | gate rows (f16) and the pooled rows of the complete pools (f16).  A cell is written once for its
//     position and read only by later positions; a pool's row is written when its last cell is appended (dsa.hpp), and
//     a query scores only the pools that end at or before it.  Rewinding to position L therefore needs nothing: the
//     cells past L are written again before anything reads them.  A parked image carries cells [0, L) and the pooled
//     rows [0, L / kpool) of each DSA layer, one ConversationKv per layer:
//         format = kGlmKvFormat | latent format, cells = L, heads = 1, head_dim = kv_lora, page_size = kpool,
//         pooled_rows = L / kpool, idx_dim;  k = the latent rows, k_scale = their scales (int8), v = the key | gate rows,
//         v_scale = empty, pooled = the pooled rows.
//     The rows are the identity layout (cell c is row c), read from the VRAM slots of a fully resident cache or from
//     the host copy of a streamed one (--kv-resident; the host copy is authoritative there), so an image restores into
//     either residency.  A restore clears each DSA layer (dsa_kv_reset: every page evicted, the pooled rows zero) and
//     writes the rows back where the session's own writers would have put them.
//
// The image's geometry key (SavedConversation::geometry) is glm_conversation_key: tagged with the architecture, so a
// Qwen image and a GLM image never validate against each other's session (conversation_state.cpp's geometry_key
// returns the same key for a glm5-next geometry; the Qwen key is unchanged).
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/glm.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace strata::core {

/// The tag of a GLM image's ConversationKv::format (the latent format in the low bits).
inline constexpr int kGlmKvFormat = 0x474C0000;   // "GL"
/// The first field of a glm5-next geometry key ("GLM5"); a Qwen key starts with n_embd.
inline constexpr int64_t kGlmGeometryTag = 0x474C4D35;

/// The geometry key of a glm5-next session: the tag, then the fields the saved state's shapes depend on.
std::array<int64_t, 18> glm_conversation_key(const ModelGeometry& g);

struct GlmConversationSizes {
    int64_t n_kda = 0, n_dsa = 0;
    size_t kda_hist = 0, kda_state = 0;   ///< bytes per KDA layer
    size_t running = 0;                   ///< every KDA layer: the checkpoint's gdn bytes
    size_t lat_row = 0, scale_row = 0, kg_row = 0, pooled_row = 0;   ///< bytes per cell (per pool for pooled_row)
    int64_t kpool = 0;
};
bool glm_conversation_sizes(const GlmSessionView& v, GlmConversationSizes& z, std::string& error);

/// The running state into `c.gdn` (the caller sets c.ids); the device is synchronized first.
bool glm_checkpoint_save(ConversationCheckpoint& c, const GlmSessionView& v, std::string& error);
bool glm_checkpoint_validate(const ConversationCheckpoint& c, const GlmSessionView& v, std::string& error);
/// Puts a checkpoint's running state back; the DSA cells below it are the caller's to guarantee (the live session's
/// own chain, or an image restored first).  Synchronizes.
bool glm_checkpoint_restore(const ConversationCheckpoint& c, const GlmSessionView& v, std::string& error);

/// Bytes a parked image of `view` takes (put()'s accounting: ids, checkpoints, the running state, the DSA rows).
bool glm_snapshot_bytes(const ConversationView& view, const GlmSessionView& v, size_t& bytes, std::string& error);
/// The whole session after view.ids (live state, the checkpoints, every DSA layer's cells [0, |ids|)).
bool glm_snapshot_save(SavedConversation& image, const ConversationView& view, const GlmSessionView& v,
                       std::string& error);
/// Checks an image against this session before anything is written (the key, the layer count, every size).
bool glm_snapshot_validate(const SavedConversation& image, const GlmSessionView& v, std::string& error);
/// Validates, then writes the image into the session: the DSA layers, then the live running state.  `invalid`: nothing
/// was written; `transfer_failed`: the session may be half written and must not be continued.
ConversationRestore glm_snapshot_restore(const SavedConversation& image, const GlmSessionView& v, std::string& error);

}  // namespace strata::core
