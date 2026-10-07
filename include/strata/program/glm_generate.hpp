// include/strata/program/glm_generate.hpp - the GLM-5.3-Flash (glm5-next) run of the `strata` program: see
// src/program/glm_generate.cpp.  generate.cpp calls it for a --native file whose architecture is glm5-next, after its
// preflight (header, geometry, tensor directory, the pack's expert spans).
#pragma once

#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::program {

struct GlmRunArgs {
    std::string pack;                  ///< --pack: tools/iq_pack.py's GLM pack
    std::string native;                ///< --native: any shard of the GGUF
    std::vector<int64_t> tokens;       ///< --tokens / --tokens-file: teacher-forced
    int64_t max_context = 4096;        ///< --max-context (raised to the token count)
    int window = 1;                    ///< --spec T: tokens per verify window (1..8)
    int64_t expert_cache = 0;          ///< --expert-cache N slots, -1 = auto
    int vram_reserve_mib = 700;        ///< --vram-reserve-mib (auto cache)
    std::string expert_profile;        ///< --expert-profile STRP (tools/make_profile.py)
    uint64_t resident_budget = 0;      ///< --resident-budget-gib, bytes
    bool resident_pin = false;         ///< page-locked RAM tier
    uint64_t resident_headroom = 8ull << 30;
    bool stats = false;                ///< --stats: greedy agreement line
    int64_t prefill_chunk = 0;         ///< --prefill N: the prompt path in chunks of N tokens (0: windows only)
    bool prefill_auto = false;         ///< --prefill auto: the default chunk (glm_prefill.hpp)
    int64_t prefill_until = 0;         ///< --prefill-until P: the prompt path reads [0, P), windows the rest (0: all
                                       ///< tokens but the last)
    int64_t max_new = 0;               ///< --max-new K: greedy tokens generated after the last one (0: none)
    // the CPU expert pool for the RAM tier's experts (glm_pool.hpp; STRATA_GLM_POOL=0 stages them to the GPU instead)
    int pool_workers = 0;              ///< --pool-workers (0: every physical core but the host's)
    bool no_host_worker = false;       ///< --no-host-worker
    strata::kernels::cpu::PoolAffinity pool_affinity = strata::kernels::cpu::PoolAffinity::All;   ///< --pool-affinity
};

/// Runs it; the process exit code.
int glm_generate(const GlmRunArgs& a);

}  // namespace strata::program
