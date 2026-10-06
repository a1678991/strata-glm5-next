// include/strata/core/glm_fetch.hpp - wave 4: the fetch side of GlmExperts' file tier.  The experts a decode layer
// misses in VRAM and in the RAM tier are read from the GGUF in place by FileExpertSource's reader (a pool of
// threads, each blob split into parts, O_DIRECT by default) into page-locked host slots, ALL of a layer's at once
// as soon as the router's ids are on the host, and GlmExperts copies each into VRAM as its read lands.
//
// THE LOOKAHEAD: before layer l's MoE block, the next MoE layer's router (F32 logits, sigmoid + exp_probs_b, top-k:
// the GPU kernels of router_sigmoid.hpp, the same arithmetic as the real router) is applied to layer l's MoE input;
// its ids are read back with layer l's own (no extra synchronization), and the predicted experts that only the file
// holds are read speculatively - behind layer l's own reads - so their bytes are on the way while layer l computes
// and layer l+1's attention runs.  When layer l+1's real ids arrive, a predicted read it needs becomes urgent, one it
// does not need is dropped if no part of it has started (else it lands in a slot and stays there as a cached blob).
// mHC mixes the streams between the two routers' inputs: the prediction is an estimate, and `stats_line` reports how
// well it matches (recall of the predicted set against the real one, every tier, and the file-tier misses it
// covered), beside the same numbers for "the previous token's experts at this layer".
//
// The slots also keep blobs after their layer (least recently used first out), which serves an expert that comes
// back within a few tokens without a read - what FileExpertSource's stage buffers did for the old path (256 of them).
// Nothing here changes which experts are computed or the bytes they are computed from.
//
// Environment (read at init):
//   STRATA_GLM_FETCH=0           the old path (GlmExperts reads each miss through FileExpertSource::blob)
//   STRATA_FETCH_THREADS=N       the reader's threads (default 16)
//   STRATA_GLM_FETCH_PART=B      bytes per part (default 2 MiB)
//   STRATA_GLM_FILE_IO=direct|pread|mmap   how a part is read (default direct)
//   STRATA_GLM_FETCH_SLOTS=N     host slots (default 256, at least 2 x window x k + 16), page-locked, allocated on use
//   STRATA_GLM_LOOKAHEAD=0|1|2   off / predict and read (default) / predict and count only
//   STRATA_GLM_LOOKAHEAD_K=K     experts predicted per token (default the model's k; at most 32)
#pragma once

#include "strata/core/expert_source.hpp"
#include "strata/core/glm.hpp"
#include "strata/kernels/router_sigmoid.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

class ExpertCache;

class GlmFetch {
public:
    struct Stats {
        int64_t calls = 0;            ///< layer calls with at least one file-tier expert
        int64_t file = 0;             ///< file-tier experts handed out
        int64_t slot_hits = 0;        ///< ... found in a slot from an earlier call (no read for this one)
        int64_t urgent = 0;           ///< urgent reads started (the layer's own misses)
        int64_t promoted = 0;         ///< ... speculative reads the layer needed while still queued or reading
        int64_t spec = 0;             ///< speculative reads started (the lookahead)
        int64_t spec_used = 0;        ///< ... needed by their layer (ready or promoted)
        int64_t spec_cancelled = 0;   ///< ... dropped before any part was read
        int64_t spec_unused = 0;      ///< ... read (or reading) but not needed by their layer
        double ms_wait = 0;           ///< the engine's time blocked on reads
        // the predictors, against each layer call's distinct experts (every tier)
        int64_t pred_calls = 0, pred_actual = 0, pred_size = 0, pred_hit = 0;
        int64_t prev_calls = 0, prev_actual = 0, prev_size = 0, prev_hit = 0;
        int64_t file_actual = 0;      ///< file-tier misses (every call)
        int64_t file_pred_actual = 0, file_pred_hit = 0, file_prev_actual = 0, file_prev_hit = 0;   ///< per predictor
    };

    GlmFetch() = default;
    ~GlmFetch();
    GlmFetch(const GlmFetch&) = delete;
    GlmFetch& operator=(const GlmFetch&) = delete;

    /// Starts the source's reader unless STRATA_GLM_FETCH=0 or the experts are not the GGUF in place (then
    /// `active()` is false and the caller reads through `blob`).  `blob_bytes`: the largest expert blob.
    bool init(const ModelGeometry& g, ExpertCache* cache, FileExpertSource* src, int max_t, uint64_t blob_bytes,
              std::string& err);
    bool active() const { return active_; }
    bool lookahead() const { return active_ && look_ != 0; }

    /// Enqueues on `stream` the router of `next` (the next MoE layer) on `x` (this layer's MoE input, T tokens) and
    /// the copy of its ids to the host; they are used after the stream's next synchronization (GlmExperts::moe's
    /// router sync), by the next `begin`.
    void predict(const GlmLayerWeights& next, const float* x, int T, void* stream);

    /// Layer l's distinct experts (`n`, first-seen order) and which of them only the file holds (`file[i]` != 0),
    /// once the router's ids are on the host: scores the predictions for layer l, starts the file tier's reads of the
    /// layer (urgent), then the lookahead's for the next layer (speculative).
    void begin(int64_t l, const int32_t* experts, const uint8_t* file, int n);
    /// The host bytes (page-locked when the slot could be) of file-tier expert `e` of the current layer, once read.
    /// nullptr (and `err`) when the read failed and the fallback copy too.
    const uint8_t* wait(int64_t l, int32_t e, std::string& err);
    /// After the copies of the layer's blobs have landed: its slots may be reused.
    void end();

    const Stats& stats() const { return st_; }
    /// "glm fetch: ..." - the reader's rate, the slots, the lookahead and the predictors.
    std::string stats_line() const;

private:
    struct Slot {
        int64_t key = -1;              ///< layer * n_expert + expert, -1 free
        uint8_t* host = nullptr;
        bool pinned = false;
        bool reading = false;          ///< a read was started and not yet seen done
        int64_t pin_call = -1;         ///< not evicted while >= the current call
        int64_t spec_call = -1;        ///< a speculative read for that call (-1: none pending)
        int64_t read_for = -1;         ///< the call the last read was started for
        uint64_t used = 0;             ///< LRU stamp
        FileExpertSource::ReadTicket t;
    };
    int alloc_slot();                  ///< a free or evictable slot (host allocated), or -1
    int issue(int64_t l, int32_t e, bool urgent, int64_t pin_call);   ///< the slot, or -1
    void free_slot(int s);

    ModelGeometry g_;
    ExpertCache* cache_ = nullptr;
    FileExpertSource* src_ = nullptr;
    bool active_ = false, cold_ = false;
    int look_ = 1;
    int max_t_ = 0, k_ = 0, kp_ = 0;
    uint64_t blob_bytes_ = 0;
    std::unique_ptr<Slot[]> slots_;
    int n_slots_ = 0;
    std::unordered_map<int64_t, int> where_;
    uint64_t stamp_ = 0;
    int64_t call_ = 0;
    // the lookahead: device logits, ids, weights; host ids (page-locked)
    strata::kernels::SigmoidRouterShape prs_;
    float* d_plog_ = nullptr;
    int32_t* d_pids_ = nullptr;
    float* d_pw_ = nullptr;
    int32_t* h_pids_ = nullptr;
    int64_t pending_layer_ = -1;       ///< predict() enqueued for this layer, not yet consumed
    int pending_t_ = 0;
    std::vector<std::vector<int32_t>> pred_;   ///< per layer: the last prediction (distinct ids)
    std::vector<std::vector<int32_t>> prev_;   ///< per layer: the last call's distinct experts
    std::vector<int> spec_slots_;              ///< slots with a speculative read pending for some call
    Stats st_;
};

}  // namespace strata::core
