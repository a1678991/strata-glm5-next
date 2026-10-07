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
    bool prefill_auto = false;         ///< --prefill auto: the largest of 8192 / 4096 / 2048 that fits (glm_generate.cpp)
    int64_t prefill_until = 0;         ///< --prefill-until P: the prompt path reads [0, P), windows the rest (0: all
                                       ///< tokens but the last)
    int64_t max_new = 0;               ///< --max-new K: greedy tokens generated after the last one (0: none)
    // the CPU expert pool for the RAM tier's experts (glm_pool.hpp; STRATA_GLM_POOL=0 stages them to the GPU instead)
    int pool_tasks = 0;                ///< --pool-tasks (0: three row tasks per participating thread)
    int pool_workers = 0;              ///< --pool-workers (0: every physical core but the host's)
    bool no_host_worker = false;       ///< --no-host-worker
    strata::kernels::cpu::PoolAffinity pool_affinity = strata::kernels::cpu::PoolAffinity::All;   ///< --pool-affinity
    int glm_pool = -1;                 ///< --glm-pool: 1 on, 0 off (staging), -1 auto (calibrated at start)
    // ---- --serve (src/program/glm_serve.cpp): the stdin/stdout protocol of generate.cpp's serve loop
    bool serve = false;                ///< --serve: stay resident and take GEN requests on stdin
    bool window_given = false;         ///< --spec T was given (serve: the teacher-forced read window; default 1)
    std::string kv = "int8";           ///< --kv int8 (production) | f16 (the parity latent, never streamed)
    int64_t kv_resident = 0;           ///< --kv-resident N: DSA cells per layer kept in VRAM (0: all)
    std::vector<int64_t> eos_ids;      ///< --eos-ids: a generated id among them ends the reply ("stop")
    int64_t turn_token = -1;           ///< --turn-token: the checkpoint goes in front of the last one in a prompt
    int64_t tail_role_token = -1;      ///< --tail-role-token: not used by GLM (said once)
    int prompt_cache = 6;              ///< --prompt-cache N: conversation checkpoints kept (0: none)
    int64_t prompt_cache_every = 16384;   ///< --prompt-cache-every N: also every N freshly read prompt tokens
    int64_t prompt_cache_root = 2048;  ///< --prompt-cache-root N: the first turn boundary of a prompt read from 0
    int64_t short_read = 64;           ///< --short-read N: a prompt part of at most N tokens goes through the windows
    int64_t conversation_cache_mib = 0;          ///< --conversation-cache-mib: RAM for parked conversations
    int conversation_cache_slots = 4;            ///< --conversation-cache-slots
    int64_t conversation_cache_min_free_mib = 2560;   ///< --conversation-cache-min-free-mib
    std::string conversation_cache_disk;         ///< --conversation-cache-disk PATH (empty: off)
    int64_t conversation_cache_disk_mib = 10240; ///< --conversation-cache-disk-mib
    bool conversation_cache_disk_keep_superseded = false;   ///< --conversation-cache-disk-keep-superseded
    std::string expert_profile_save;   ///< --expert-profile-save: not used by GLM (no adaptive tier; said once)
};

/// Runs it; the process exit code.
int glm_generate(const GlmRunArgs& a);

}  // namespace strata::program
