// include/strata/core/glm_seldump.hpp - GLM-5.3-Flash: the DSA selection of chosen positions, written for the
// selection-overlap check against llama.cpp (~/dev/strata-glm/validate: glm-splice --sel-dump, scripts/sel_overlap.py).
//
//   STRATA_GLM_SEL_DUMP=<dir>      on (off when unset or empty: no effect on anything else)
//   STRATA_GLM_SEL_AT=p1,p2,...    the positions (0-based sequence positions)
//
// For every DSA layer and listed position, whichever path computes it (the decode windows, GlmEngine::layer, or the
// prompt path, GlmPrefill), one text file <dir>/sel-L<layer>-P<pos>.txt:
//
//   glm-sel 1
//   src strata-decode | strata-prefill
//   layer <l>
//   pos <p>
//   n_bid <(p + 1) / kpool: the pools that end at or before p>
//   width <the cells attended: min(n_bid, idx_top_k / kpool) * kpool + (p + 1) % kpool>
//   ids <width cell ids: the selected pools' cells ascending, then the tail> (cell = position: one sequence from 0)
//   scores <n_bid floats: the indexer score of every pool j < n_bid, in pool order>   (absent when not computed:
//          the prompt path skips the indexer while the selection is the identity, n_bid <= 512)
//
// glm-splice --sel-dump writes the same format from llama.cpp's indexer_top_k-N / indexer_score-N.
#pragma once

#include "strata/kernels/dsa.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct GlmSelDump {
    bool on = false;
    std::string dir;
    std::vector<int64_t> at;   ///< sorted, unique

    /// The environment's settings, parsed once.
    static const GlmSelDump& get();
    /// Whether `pos` is listed.
    bool has(int64_t pos) const;
    /// Whether a listed position lies in [p0, p0 + n).
    bool any_in(int64_t p0, int64_t n) const;
    /// Writes the file of (layer, pos) from DEVICE rows: `ids` the selection (dsa_sel_width(pos) cell ids; null = the
    /// identity 0..pos) and `scores` the pool scores (n_bid floats; null = not computed). Synchronizes `stream`.
    bool write(int64_t layer, int64_t pos, const int32_t* ids, const float* scores, const kernels::DsaShapes& s,
               void* stream, const char* src, std::string& err) const;
};

}  // namespace strata::core
