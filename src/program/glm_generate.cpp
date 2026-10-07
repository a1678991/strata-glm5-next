// src/program/glm_generate.cpp - `strata --pack <glm pack> --native <glm5-next gguf> --tokens-file ids ...`: the
// GLM-5.3-Flash decode path, teacher-forced through verify windows (include/strata/core/glm.hpp).
//
// What it does, in order: the model's header and geometry (generate.cpp's glm5_next_preflight has checked the tensor
// directory and the pack's expert layout), the token embedding in mapped host memory (NativeEmbed), the pack's F32
// rows into a device arena, every quantized non-expert tensor from the GGUF (NativeDense), the output head
// (NativeHead) when every layer runs, the expert tiers when an MoE layer runs (a VRAM cache filled in the profile's
// order, --resident-budget-gib of page-locked RAM, the GGUF shards in place for the rest), then the prompt through
// windows of --spec T tokens (default 1), each committed whole.
//
// Measurement hooks:
//   STRATA_LOGPOS=<file>      the teacher-forced top-k rows of every position (Verifier::window_logprobs' format;
//                             STRATA_LOGPOS_TOPK=20, STRATA_LOGPOS_EXTRA as on the Qwen path)
//   STRATA_GLM_DUMP=<dir>     <dir>/l_out-<l>.glmt: the 4 hyper-connection streams after layer l, one row per token
//                             (validate/scripts/glmt.py; glm-splice --replace l_out-<l>=<file>)
//   STRATA_GLM_DUMP_LAYERS=a,b,...   only these layers (default: every layer that runs)
//   STRATA_GLM_DUMP_FROM=P    only the rows of positions >= P (the prompt path's too)
//   STRATA_GLM_LAYERS=N       run layers 0..N-1 only (no head unless N is the whole model): a stage's check
//   STRATA_GLM_FROM=i, STRATA_GLM_INPUT=<file>   layers i..N-1 only, on the rows of <file> (llama.cpp's l_out-(i-1))
//   STRATA_GLM_KV=f16         the DSA latent in f16 (the parity layout) instead of int8
//   STRATA_GLM_LOGITS=<file> every position's full logits (GLMT, glm-splice --logits' format); with
//                             STRATA_GLM_LOGITS_FROM=P only positions >= P
//   STRATA_GLM_PROFILE=1      per-layer GPU times
//   STRATA_GLM_NODES=<dir>, STRATA_GLM_NODES_LAYER=l   the intermediate nodes of layer l (glm.hpp set_node_dump),
//                             named as llama.cpp's, for glm-splice --dump of the same names
//   STRATA_GLM_MARK_EVERY=N, STRATA_GLM_MARKS=a,b,...   a timing line at every N-th token and at the listed
//                             positions (the first window that reaches each): seconds since the first window, and the
//                             tier counters so far (cumulative; differences between two marks give a segment's rate)
//
// The prompt path (--prefill N|auto, port plan Stage 6; include/strata/prefill/glm_prefill.hpp): positions
// [0, P) are read in chunks of N tokens (P = --prefill-until, default every token but the last), the rest through
// the windows as above, teacher-forced; --max-new K then generates K greedy tokens (one window each) and prints
// them as "GLM_GEN <ids>".
//   STRATA_GLM_PREFILL_LOGPOS=L   STRATA_LOGPOS also gets the prompt's last L positions (1..max window, default 1:
//                             the position whose logits pick the first token after the prompt)
//   STRATA_GLM_STATE=<dir>    the session after position P (GlmEngine::dump_state): KDA history and state, DSA
//                             latents, key | gate rows and pooled rows; STRATA_GLM_STATE_AT=P sets P for a run
//                             without the prompt path (the windows then stop at P once), to compare the two
//   STRATA_GLM_PREFILL_TIMING=1   the prompt path's GPU time per phase
//
// The RAM tier's experts are computed by the CPU expert pool (include/strata/core/glm_pool.hpp; --pool-workers,
// --pool-affinity, --no-host-worker as on the Qwen path; the host thread is pinned to the first physical core while the
// windows run) or copied into VRAM staging and computed on the GPU: --glm-pool on|off|auto (default auto: a calibration
// at start times both on the same RAM-tier experts and keeps the faster, GlmExperts::calibrate_pool;
// STRATA_GLM_POOL_CALIBRATE=N experts, default 256).  STRATA_GLM_POOL=0|1 overrides --glm-pool (the A/B arms).
#include "strata/program/glm_generate.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/glm.hpp"
#include "strata/core/glm_experts.hpp"
#include "strata/core/glm_pool.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/glm_prefill.hpp"
#include "strata/program/glm_serve.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace strata::program {
namespace {

using Clock = std::chrono::steady_clock;
namespace sc = strata::core;

int fail(const std::string& msg) {
    std::fprintf(stderr, "strata generate: glm5-next: %s\n", msg.c_str());
    return 1;
}

double free_mib() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return (double) f / 1048576.0;
}

// --prefill auto: the chunk sizes it tries, largest first.  Measured on the RTX 3090 (UD-IQ3_XXS, 16,376-token
// prompt, window 2): 8,192 read at 281.9 tok/s with 2,862 MiB of buffers, 4,096 at 155.4 with 1,606.
constexpr int64_t kAutoChunks[] = {8192, 4096, 2048};

// With --expert-cache auto the prompt buffers come out of the expert cache's VRAM: the largest chunk whose buffers
// take at most this share of what the two share (default 25%; STRATA_GLM_PREFILL_SHARE=<percent>).  On the 3090
// 8,192's buffers are 18.6% of it and cost the cache ~147 of ~1,600 experts (~2% of the lookups move to RAM).
double auto_share() {
    const char* v = std::getenv("STRATA_GLM_PREFILL_SHARE");
    return v != nullptr && *v ? std::max(0.0, std::min(100.0, std::atof(v))) / 100.0 : 0.25;
}

// The largest of kAutoChunks (each capped at the prompt) whose buffers fit `limit` MiB, else the smallest; logged.
int64_t auto_chunk(const sc::ModelGeometry& g, const sc::GlmConfig& cfg, int64_t P, double limit, double room,
                   const std::string& what) {
    std::string tried;
    int64_t pick = 0;
    double pick_mib = 0;
    for (const int64_t c : kAutoChunks) {
        const double mib = (double) strata::prefill::GlmPrefill::bytes_needed(g, cfg, std::min(c, P)) / 1048576.0;
        char b[64];
        std::snprintf(b, sizeof b, "%s%lld: %.0f MiB", tried.empty() ? "" : ", ", (long long) c, mib);
        tried += b;
        if (pick == 0 && mib <= limit) { pick = c; pick_mib = mib; }
    }
    const bool fits = pick != 0;
    if (!fits) {
        pick = kAutoChunks[sizeof kAutoChunks / sizeof kAutoChunks[0] - 1];
        pick_mib = (double) strata::prefill::GlmPrefill::bytes_needed(g, cfg, std::min(pick, P)) / 1048576.0;
    }
    std::fprintf(stderr, "strata generate: glm5-next: --prefill auto: chunks of %lld tokens (%.0f MiB of buffers, %.1f%% "
                         "%s%s; %s)\n",
                 (long long) std::min(pick, P), pick_mib, room > 0 ? 100.0 * pick_mib / room : 0.0, what.c_str(),
                 fits ? "" : "; none fits, the smallest taken", tried.c_str());
    return pick;
}

// What a configuration needs in VRAM besides the expert cache, said when it does not fit (wave 6: --max-context and
// --kv-resident decide the DSA latent cache, which the session holds; the prompt path's buffers come on top).
std::string vram_shortfall(const sc::ModelGeometry& g, const sc::GlmConfig& cfg, double session_mib, double staging_mib,
                           double prompt_mib, int reserve_mib, double free) {
    const int64_t slots = sc::glm_kv_slots(g, cfg);
    const int64_t cells = slots > 0 ? cfg.kv_resident : cfg.max_cells;
    const double dsa_mib = (double) g.n_qsa_layers() * (double) cells *
                           (double) (g.kv_lora + g.kv_lora / 64 * 4 + 4 * g.idx_key_dim + 2 * g.idx_key_dim / std::max<int64_t>(1, g.idx_kpool)) /
                           1048576.0;
    char b[768];
    std::snprintf(b, sizeof b, "not enough VRAM: %.0f MiB free; the session needs %.0f MiB (context %lld, the DSA latent "
                               "cache %.0f MiB of it with %s), the expert staging %.0f MiB, the prompt path's buffers "
                               "%.0f MiB, the reserve %d MiB + 512 MiB.  Lower --max-context, keep fewer DSA cells in "
                               "VRAM (--kv-resident N streams the rest from page-locked RAM), or a smaller --prefill "
                               "chunk", free, session_mib, (long long) cfg.max_cells, dsa_mib,
                 slots > 0 ? ("--kv-resident " + std::to_string((long long) cfg.kv_resident)).c_str()
                           : "every cell resident", staging_mib, prompt_mib, reserve_mib);
    return b;
}

}  // namespace

int glm_generate(const GlmRunArgs& a) {
    std::string err;
    const auto t_start = Clock::now();
    std::vector<std::string> shards;
    sc::ModelGeometry g;
    try {
        shards = strata::gguf_split_paths(a.native);
        const strata::GgufModel model(shards);
        if (!sc::geometry_from_gguf(model.meta(), g, err)) return fail(err);
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    if (a.tokens.size() < 1 && !a.serve) return fail("no tokens (--tokens / --tokens-file)");
    for (const int64_t t : a.tokens)
        if (t < 0 || t >= g.n_vocab) return fail("token id " + std::to_string(t) + " is outside the vocabulary");
    int64_t n_run = g.n_layers;
    if (const char* v = std::getenv("STRATA_GLM_LAYERS"); v != nullptr && *v) n_run = std::atoll(v);
    if (n_run < 1 || n_run > g.n_layers) return fail("STRATA_GLM_LAYERS must be 1.." + std::to_string(g.n_layers));
    const bool full = n_run == g.n_layers;
    const int T = std::max(1, std::min(8, a.window));
    const int64_t max_cells = a.serve ? a.max_context
                                      : std::max<int64_t>(a.max_context, (int64_t) a.tokens.size() + std::max<int64_t>(0, a.max_new));
    if (a.serve && (max_cells < 16 || max_cells > INT32_MAX / 4)) return fail("--max-context out of range");
    // the prompt path: positions [0, P) in chunks (--prefill N, or auto: the largest of kAutoChunks that fits, chosen
    // below where the VRAM is budgeted: with --expert-cache auto before the cache is sized, else from the VRAM left
    // once the session is in)
    const int64_t n_tok = (int64_t) a.tokens.size();
    int64_t pf_chunk = a.prefill_auto ? kAutoChunks[0] : a.prefill_chunk;
    bool pf_chosen = !a.prefill_auto;
    int64_t P = 0;
    if (pf_chunk > 0 && a.serve) {
        P = max_cells;   // --serve: any prompt up to the context; the chunk is chosen for the longest
    } else if (pf_chunk > 0) {
        P = a.prefill_until > 0 ? std::min<int64_t>(a.prefill_until, n_tok) : std::max<int64_t>(1, n_tok - 1);
        if (std::getenv("STRATA_GLM_FROM") != nullptr) return fail("STRATA_GLM_FROM does not combine with --prefill");
    }
    const bool any_moe = n_run > g.n_dense_lead;
    // the session's configuration (the expert cache's sizing below counts exactly this session)
    sc::GlmConfig cfg;
    cfg.max_t = T;
    cfg.max_cells = max_cells;
    cfg.n_run = n_run;
    if (const char* v = std::getenv("STRATA_GLM_KV"); v != nullptr && std::string(v) == "f16")
        cfg.kv_fmt = strata::kernels::kDsaLatF16;
    if (a.kv == "f16") cfg.kv_fmt = strata::kernels::kDsaLatF16;
    else if (a.kv != "int8" && !a.kv.empty()) return fail("--kv " + a.kv + ": glm5-next keeps its DSA latent as int8 (or f16)");
    cfg.kv_resident = a.kv_resident;
    if (sc::glm_kv_slots(g, cfg) > 0 && cfg.kv_fmt != strata::kernels::kDsaLatInt8)
        return fail("--kv-resident streams the int8 latent only (not --kv f16)");
    if (sc::glm_kv_slots(g, cfg) > 0 && cfg.kv_resident < sc::glm_kv_resident_min(g, T))
        return fail("--kv-resident " + std::to_string((long long) cfg.kv_resident) + " is below the " +
                    std::to_string((long long) sc::glm_kv_resident_min(g, T)) + " cells a streamed cache needs (a window of " +
                    std::to_string(T) + " token(s): its selections resident at once, and at least 1,024 pages)");
    if (!strata::kernels::cpu::expert_layout_load(a.pack, g.n_layers, g.n_expert, err, g.n_embd, g.n_ff,
                                                  g.n_dense_lead))
        return fail(a.pack + ": " + err);

    // ---- the dense weights: F32 rows from the pack, quantized tensors from the GGUF, the embedding in host memory
    sc::NativeEmbed embed;
    if (!embed.load(shards, g.n_embd, g.n_vocab, err)) return fail(err);
    std::set<std::string> skip;
    if (!sc::NativeDense::served_names(shards, false, skip, err)) return fail(err);
    skip.insert("output.weight");
    skip.insert("token_embd.weight");
    uint64_t pool = 0;
    if (!sc::WeightTable::pool_bytes(a.pack, pool, err, &skip)) return fail(err);
    void* arena = nullptr;
    if (cudaMalloc(&arena, std::max<uint64_t>(pool, 256)) != cudaSuccess) return fail("cudaMalloc of the F32 arena");
    std::unique_ptr<void, void (*)(void*)> arena_guard(arena, [](void* p) { cudaFree(p); });
    sc::WeightTable wt;
    if (!wt.load(a.pack, arena, std::max<uint64_t>(pool, 256), err, &skip)) return fail(err);
    sc::NativeDense dense;
    if (!dense.load(shards, wt, err, false, 0, g.n_layers)) return fail(err);
    std::fprintf(stderr, "strata generate: glm5-next: %.2f MiB of F32 rows from %s, %zu native matrices (%.2f MiB) from "
                         "the GGUF, token embedding %s in mapped host memory (%.0f MiB)\n",
                 (double) pool / 1048576.0, a.pack.c_str(), dense.tensor_count(),
                 (double) dense.weight_bytes() / 1048576.0, strata::ggml_type_name((uint32_t) embed.type()),
                 (double) embed.bytes() / 1048576.0);
    sc::NativeHead head;
    if (full) {
        if (!head.load(shards, g.n_embd, g.n_vocab, err)) return fail(err);
        std::fprintf(stderr, "strata generate: glm5-next: output head %s, %.2f MiB\n",
                     strata::ggml_type_name((uint32_t) head.type()), (double) head.weight_bytes() / 1048576.0);
    }

    // ---- the expert tiers
    sc::FileExpertSource src;
    sc::ExpertCache cache;
    sc::GlmExperts experts;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> cpu_pool;
    std::unique_ptr<sc::GlmPool> glm_pool = std::make_unique<sc::GlmPool>();   // reset when --glm-pool auto picks staging
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (any_moe) {
        src.set_gguf(a.native);
        if (!src.open(a.pack, g.n_layers, g.n_expert, err)) return fail(err);
        if (!a.expert_profile.empty()) {
            int64_t pslots = 0;
            if (!sc::read_expert_profile(a.expert_profile, g.n_layers, g.n_expert, ranked, pslots, err))
                return fail(err);
        } else {   // no profile: layer-interleaved index order
            for (int64_t e = 0; e < g.n_expert; ++e)
                for (int64_t l = g.n_dense_lead; l < g.n_layers; ++l) ranked.emplace_back((int32_t) l, (int32_t) e);
        }
        const auto& lay = strata::kernels::cpu::expert_layout();
        // the VRAM cache: --expert-cache N slots, or "auto" (-1): what the card has left above --vram-reserve-mib
        // once the session and the staging buffers are counted (estimated here, before they exist)
        int64_t want = a.expert_cache;
        if (want < 0) {
            sc::GlmConfig probe = cfg;
            const double session = (double) sc::glm_session_bytes(g, probe) / 1048576.0;
            // the staging slots, and (wave 4) the lookahead's VRAM slots when STRATA_GLM_PRESTAGE asks for them
            const char* pre = std::getenv("STRATA_GLM_PRESTAGE");
            const int64_t n_pre = pre != nullptr ? std::max<int64_t>(0, std::min<int64_t>(64, std::atoll(pre))) : 0;
            const double staging = (double) (T * g.n_expert_used + n_pre) * (double) lay.max_blob / 1048576.0;
            double room = free_mib() - session - staging - 512.0 - (double) a.vram_reserve_mib;
            if (room < 0)   // wave 6: say what does not fit instead of failing later in an allocation
                return fail(vram_shortfall(g, cfg, session, staging, 0.0, a.vram_reserve_mib, free_mib()));
            if (P > 0 && !pf_chosen) {   // --prefill auto: the chunk first, then the cache gets the rest
                // (the share, or whatever the cache cannot use once every expert is in it)
                double all = 0;
                for (const auto& [l, e] : ranked) all += (double) lay.blob_bytes(l) / 1048576.0;
                pf_chunk = auto_chunk(g, probe, P, std::max(auto_share() * room, room - all), room, "of the " +
                                      std::to_string((long long) room) + " MiB the expert cache and the prompt "
                                      "buffers share; at most " + std::to_string((int) (100 * auto_share())) +
                                      "% or what every expert (" + std::to_string((long long) all) + " MiB) leaves");
                pf_chosen = true;
            }
            const double prompt = P > 0 ? (double) strata::prefill::GlmPrefill::bytes_needed(
                                              g, probe, std::min<int64_t>(pf_chunk, P)) / 1048576.0 : 0.0;
            room -= prompt;
            if (room < 0)
                return fail(vram_shortfall(g, cfg, session, staging, prompt, a.vram_reserve_mib, free_mib()));
            want = 0;
            for (const auto& [l, e] : ranked) {
                const double b = (double) lay.blob_bytes(l) / 1048576.0;
                if (room < b) break;
                room -= b;
                ++want;
            }
        }
        want = std::min<int64_t>(want, (int64_t) ranked.size());
        if (want > 0) {
            std::vector<int64_t> sized;
            for (int64_t i = 0; i < want; ++i) sized.push_back((int64_t) lay.blob_bytes(ranked[(size_t) i].first));
            if (!cache.open_sized(sized, g.n_layers, g.n_expert, err)) return fail(err);
            for (int64_t i = 0; i < want; ++i) {
                const auto [l, e] = ranked[(size_t) i];
                const int32_t slot = cache.admit(l, e);
                const uint8_t* b = src.blob(l, e);
                if (slot == sc::kNotResident || b == nullptr ||
                    !cache.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(l)))
                    return fail("filling the expert cache: " + err);
            }
            std::fprintf(stderr, "strata generate: glm5-next: expert cache %lld of %lld experts in VRAM (%.2f GiB), in "
                                 "%s order\n",
                         (long long) cache.resident(), (long long) ((g.n_layers - g.n_dense_lead) * g.n_expert),
                         cache.gib(), a.expert_profile.empty() ? "index" : "profile");
        }
        if (a.resident_budget > 0) {
            if (!cache.valid()) return fail("--resident-budget-gib needs a VRAM expert cache (--expert-cache N >= 1)");
            if (!src.pin_cache_complement(cache, err, a.resident_pin, {}, -1, a.resident_headroom, a.resident_budget,
                                          &ranked))
                return fail(err);
            std::fprintf(stderr, "strata generate: glm5-next: RAM tier %.2f GiB (%s), the rest from the file tier\n",
                         (double) src.resident_bytes() / 1073741824.0,
                         src.complement_pinned() ? "page-locked" : "pageable");
        }
        // --glm-pool on|off|auto (1 / 0 / -1); STRATA_GLM_POOL=0|1 overrides it
        int pool_mode = a.glm_pool;
        if (const char* pv = std::getenv("STRATA_GLM_POOL"); pv != nullptr && *pv) pool_mode = std::atoi(pv) != 0 ? 1 : 0;
        if (src.resident_bytes() > 0 && pool_mode != 0) {
            cpu_pool = std::make_unique<strata::kernels::cpu::ExpertPool>(a.pool_workers, /*pin=*/true,
                                                                          /*host_works=*/!a.no_host_worker,
                                                                          a.pool_affinity);
            if (!glm_pool->init(g, cpu_pool.get(), T, (int) g.n_expert_used, err)) return fail(err);
            experts.set_pool(glm_pool.get());
            std::fprintf(stderr, "strata generate: glm5-next: RAM-tier experts on the CPU pool, %d threads (%d workers%s)\n",
                         glm_pool->workers(), cpu_pool->workers(), cpu_pool->host_works() ? " + the host thread" : "");
        }
        if (!experts.init(g, cache.valid() ? &cache : nullptr, &src, T, err)) return fail(err);
        if (glm_pool->ready() && pool_mode < 0) {
            // auto: the pool against staging + the GPU on the same RAM-tier experts (glm_experts.hpp
            // calibrate_pool), the faster kept.  On epyc (23 Zen 4 workers, IQ2_S) the pool read 16.6 GB/s against
            // 26.9 GB/s over PCIe; on the 7950X with glm-mini it was 3x faster than staging.
            int n_cal = 256;
            if (const char* v = std::getenv("STRATA_GLM_POOL_CALIBRATE"); v != nullptr && *v)
                n_cal = std::max(1, std::atoi(v));
            sc::GlmExperts::PoolCalibration c;
            const auto tc = Clock::now();
            if (!experts.calibrate_pool(n_cal, c, err)) return fail(err);
            const bool keep = c.ms_pool <= c.ms_gpu;
            std::fprintf(stderr, "strata generate: glm5-next: --glm-pool auto: %d RAM-tier experts from layer %lld on "
                                 "(%d calls of %d tokens, %.1f MB): CPU pool %.1f ms (%.1f GB/s), staging + GPU %.1f ms "
                                 "(%.1f GB/s): %s (calibration %.2f s)\n",
                         c.experts, (long long) c.layer, c.calls, T, (double) c.bytes / 1e6, c.ms_pool,
                         (double) c.bytes / 1e6 / std::max(c.ms_pool, 1e-6), c.ms_gpu,
                         (double) c.bytes / 1e6 / std::max(c.ms_gpu, 1e-6),
                         keep ? "the CPU pool" : "staging, the pool stopped",
                         std::chrono::duration<double>(Clock::now() - tc).count());
            if (!keep) {
                experts.drop_pool();
                glm_pool.reset();
                cpu_pool.reset();
            }
        }
    }

    // ---- the session
    sc::GlmEngine eng;
    if (!eng.init(wt, g, embed, full ? &head : nullptr, any_moe ? &experts : nullptr, cfg, err)) {
        cudaGetLastError();
        return fail(err + " (" + vram_shortfall(g, cfg, (double) sc::glm_session_bytes(g, cfg) / 1048576.0, 0.0, 0.0,
                                                0, free_mib()) + ")");
    }
    std::fprintf(stderr, "strata generate: glm5-next: session %.1f MiB (window %d, context %lld, DSA latent %s%s), "
                         "layers 0-%lld%s; %.0f MiB of VRAM free\n",
                 (double) eng.session_bytes() / 1048576.0, T, (long long) max_cells,
                 cfg.kv_fmt == strata::kernels::kDsaLatF16 ? "f16" : "int8",
                 sc::glm_kv_slots(g, cfg) > 0
                     ? (", " + std::to_string((long long) cfg.kv_resident) + " cells per layer in VRAM, " +
                        std::to_string((long long) (eng.session_host_bytes() >> 20)) + " MiB page-locked").c_str()
                     : "",
                 (long long) n_run - 1, full ? " and the head" : " (no head: STRATA_GLM_LAYERS)", free_mib());

    strata::prefill::GlmPrefill prefill;
    if (P > 0 && !pf_chosen) {   // --prefill auto without --expert-cache auto: what the session and the cache left
        const double room = free_mib() - 512.0;
        pf_chunk = auto_chunk(g, cfg, P, room, room, "of the " + std::to_string((long long) room) +
                              " MiB of VRAM free above a 512 MiB margin");
        pf_chosen = true;
    }
    if (P > 0) {
        if (!prefill.init(eng, cache.valid() ? &cache : nullptr, any_moe ? &src : nullptr, std::min<int64_t>(pf_chunk, P),
                          err)) {
            cudaGetLastError();
            return fail(err + " (" + vram_shortfall(g, cfg, 0.0, 0.0, (double) strata::prefill::GlmPrefill::bytes_needed(
                                                    g, cfg, std::min<int64_t>(pf_chunk, P)) / 1048576.0, 0, free_mib()) +
                        ")");
        }
        std::fprintf(stderr, "strata generate: glm5-next: prompt path: positions 0-%lld in chunks of %lld tokens "
                             "(%.1f MiB of buffers), the rest through windows; %.0f MiB of VRAM free\n",
                     (long long) P - 1, (long long) prefill.chunk(), (double) prefill.device_bytes() / 1048576.0,
                     free_mib());
    }

    // ---- --serve (wave 6): the persistent engine takes it from here (src/program/glm_serve.cpp)
    if (a.serve) {
        if (!full) return fail("--serve needs the whole model (unset STRATA_GLM_LAYERS)");
        if (P == 0)
            std::fprintf(stderr, "strata serve: glm5-next: no --prefill: every prompt token goes through the windows\n");
        struct ServePin {
            strata::kernels::cpu::ThreadAffinity prev;
            ~ServePin() { if (prev.valid) strata::kernels::cpu::restore_thread_affinity(prev); }
        } serve_pin;
        if (glm_pool && glm_pool->ready()) {
            const std::vector<int> cores = strata::kernels::cpu::physical_cores(false);
            if (!cores.empty()) serve_pin.prev = strata::kernels::cpu::pin_current_thread(cores[0]);
        }
        GlmServeParts parts;
        parts.args = &a;
        parts.g = &g;
        parts.shards = shards;
        parts.eng = &eng;
        parts.prefill = P > 0 ? &prefill : nullptr;
        parts.experts = any_moe ? &experts : nullptr;
        parts.src = any_moe ? &src : nullptr;
        parts.cache = cache.valid() ? &cache : nullptr;
        parts.window = T;
        parts.pool_workers = glm_pool && glm_pool->ready() ? glm_pool->workers() : 0;
        const double load_s = std::chrono::duration<double>(Clock::now() - t_start).count();
        std::fprintf(stderr, "strata generate: glm5-next: loaded in %.1f s; serving\n", load_s);
        return glm_serve(parts);
    }

    // ---- the dumps
    std::vector<std::unique_ptr<sc::GlmtWriter>> writers((size_t) g.n_layers);
    if (const char* dir = std::getenv("STRATA_GLM_DUMP"); dir != nullptr && *dir) {
        std::vector<int64_t> layers;
        if (const char* v = std::getenv("STRATA_GLM_DUMP_LAYERS"); v != nullptr && *v) {
            std::stringstream ss(v);
            std::string item;
            while (std::getline(ss, item, ',')) layers.push_back(std::atoll(item.c_str()));
        } else {
            for (int64_t l = 0; l < n_run; ++l) layers.push_back(l);
        }
        for (const int64_t l : layers) {
            if (l < 0 || l >= n_run) return fail("STRATA_GLM_DUMP_LAYERS names layer " + std::to_string(l));
            writers[(size_t) l] = std::make_unique<sc::GlmtWriter>();
            const std::string name = "l_out-" + std::to_string(l);
            if (!writers[(size_t) l]->create(std::string(dir) + "/" + name + ".glmt", name, g.n_embd, g.hc, err))
                return fail(err);
            eng.set_dump(l, writers[(size_t) l].get());
            prefill.set_dump(l, writers[(size_t) l].get());
        }
    }
    if (const char* dir = std::getenv("STRATA_GLM_NODES"); dir != nullptr && *dir) {
        const char* v = std::getenv("STRATA_GLM_NODES_LAYER");
        const int64_t l = v != nullptr ? std::atoll(v) : 0;
        if (l < 0 || l >= n_run) return fail("STRATA_GLM_NODES_LAYER names a layer that does not run");
        if (!eng.set_node_dump(l, dir, err)) return fail(err);
    }
    std::FILE* logpos = nullptr;
    if (const char* p = std::getenv("STRATA_LOGPOS"); p != nullptr && *p) {
        if (!full) return fail("STRATA_LOGPOS needs the whole model (unset STRATA_GLM_LAYERS)");
        logpos = std::fopen(p, "ab");
        if (logpos == nullptr) return fail(std::string("cannot open ") + p);
    }
    const int32_t extra = [] {
        const char* p = std::getenv("STRATA_LOGPOS_EXTRA");
        return p != nullptr ? (int32_t) std::atoi(p) : (int32_t) 248046;
    }();

    // STRATA_GLM_FROM=i STRATA_GLM_INPUT=<llama.cpp's l_out-(i-1).glmt>: layers [i, n_run) on the reference's input
    int64_t from = 0;
    std::FILE* x_file = nullptr;
    std::vector<float> x_in;
    if (const char* v = std::getenv("STRATA_GLM_FROM"); v != nullptr && *v) {
        from = std::atoll(v);
        const char* in = std::getenv("STRATA_GLM_INPUT");
        if (from < 1 || from >= n_run || in == nullptr) return fail("STRATA_GLM_FROM needs 1..n_run-1 and STRATA_GLM_INPUT");
        x_file = std::fopen(in, "rb");
        char hdr[128];
        uint64_t row_elems = 0, rows = 0;
        if (x_file == nullptr || std::fread(hdr, 1, 128, x_file) != 128 || std::string(hdr, 4) != "GLMT")
            return fail(std::string("cannot read the GLMT input ") + in);
        std::memcpy(&row_elems, hdr + 16, 8);
        std::memcpy(&rows, hdr + 24, 8);
        if (row_elems != (uint64_t) (g.hc * g.n_embd) || rows < a.tokens.size())
            return fail(std::string(in) + ": rows of the wrong size or too few");
        x_in.resize((size_t) T * g.hc * g.n_embd);
    }
    // STRATA_GLM_LOGITS=<file>: every position's full logits (GLMT, as glm-splice --logits), for a full-vocab KLD
    std::unique_ptr<sc::GlmtWriter> logits_out;
    int64_t logits_from = 0;   // STRATA_GLM_LOGITS_FROM=P: only the rows of positions >= P
    if (const char* v = std::getenv("STRATA_GLM_LOGITS_FROM"); v != nullptr && *v) logits_from = std::atoll(v);
    std::vector<float> lrow;
    if (const char* p = std::getenv("STRATA_GLM_LOGITS"); p != nullptr && *p) {
        if (!full) return fail("STRATA_GLM_LOGITS needs the whole model");
        logits_out = std::make_unique<sc::GlmtWriter>();
        if (!logits_out->create(p, "logits", g.n_vocab, 1, err)) return fail(err);
        lrow.resize((size_t) g.n_vocab);
    }
    // ---- the prompt: [0, P) through the prompt path, the rest teacher-forced through the windows
    const int64_t n = (int64_t) a.tokens.size();
    std::vector<int32_t> win((size_t) T), nxt((size_t) T);
    const double load_s = std::chrono::duration<double>(Clock::now() - t_start).count();
    std::fprintf(stderr, "strata generate: glm5-next: loaded in %.1f s; reading %lld tokens (%lld by the prompt path) "
                         "in windows of %d\n", load_s, (long long) n, (long long) P, T);
    const char* state_dir = std::getenv("STRATA_GLM_STATE");
    int64_t state_at = P > 0 ? P : -1;
    if (const char* v = std::getenv("STRATA_GLM_STATE_AT"); v != nullptr && *v) state_at = std::atoll(v);
    if (state_dir != nullptr && *state_dir && (state_at < 1 || state_at > n))
        return fail("STRATA_GLM_STATE needs a position 1..n (STRATA_GLM_STATE_AT or --prefill)");
    auto dump_state_at = [&](int64_t q) -> bool {
        if (state_dir == nullptr || !*state_dir || q != state_at) return true;
        if (!eng.dump_state(state_dir, q, err)) return false;
        std::fprintf(stderr, "strata generate: glm5-next: session state after %lld positions written to %s\n",
                     (long long) q, state_dir);
        return true;
    };
    // STRATA_GLM_MARK_EVERY / STRATA_GLM_MARKS: timing lines while the tokens are read (no effect on the numbers)
    std::vector<int64_t> marks;
    if (const char* v = std::getenv("STRATA_GLM_MARKS"); v != nullptr && *v) {
        std::stringstream ss(v);
        std::string item;
        while (std::getline(ss, item, ',')) if (!item.empty()) marks.push_back(std::atoll(item.c_str()));
    }
    if (const char* v = std::getenv("STRATA_GLM_MARK_EVERY"); v != nullptr && std::atoll(v) > 0)
        for (int64_t m = std::atoll(v); m < n; m += std::atoll(v)) marks.push_back(m);
    std::sort(marks.begin(), marks.end());
    size_t next_mark = 0;
    int64_t agree = 0, rows = 0;
    int last_rows = 0;   // rows of the last logits written (the window's, or the prompt path's head)
    double prefill_s = 0.0;
    if (P > 0) {
        const auto tp = Clock::now();
        std::vector<int32_t> ids((size_t) P);
        for (int64_t i = 0; i < P; ++i) ids[(size_t) i] = (int32_t) a.tokens[(size_t) i];
        int n_logits = 1;
        if (const char* v = std::getenv("STRATA_GLM_PREFILL_LOGPOS"); v != nullptr && *v)
            n_logits = std::max(1, std::min(T, std::atoi(v)));
        n_logits = (int) std::min<int64_t>(n_logits, P);
        if (!prefill.run(ids.data(), P, 0, full ? n_logits : 0, err)) return fail(err);
        last_rows = n_logits;
        prefill_s = std::chrono::duration<double>(Clock::now() - tp).count();
        if (full) {   // the prompt's last positions: the rows the windows would have written for them
            const int64_t p0 = P - n_logits;
            for (int t = 0; t < n_logits; ++t) nxt[(size_t) t] = p0 + t + 1 < n ? (int32_t) a.tokens[(size_t) (p0 + t + 1)] : -1;
            if (logpos != nullptr && !eng.window_logprobs(nxt.data(), n_logits, p0, extra, logpos, err)) return fail(err);
            for (int t = 0; logits_out && t < n_logits; ++t)
                if (p0 + t >= logits_from && (!eng.copy_logits(t, lrow.data()) || !logits_out->append(lrow.data(), 1)))
                    return fail("logits dump");
            if (a.stats) {
                int32_t am[8];
                if (!eng.argmax(n_logits, am)) return fail("argmax");
                for (int t = 0; t < n_logits; ++t)
                    if (nxt[(size_t) t] >= 0) { agree += am[t] == nxt[(size_t) t]; ++rows; }
            }
        }
        std::fprintf(stderr, "strata generate: glm5-next: prompt path: %lld tokens in %.2f s: %.1f tok/s\n",
                     (long long) P, prefill_s, (double) P / prefill_s);
        if (!dump_state_at(P)) return fail(err);
    }
    // the pool's workers stay off the first physical core; the host loop takes it (session.cpp's SessionLoopScratch)
    struct HostPin {
        strata::kernels::cpu::ThreadAffinity prev;
        ~HostPin() { if (prev.valid) strata::kernels::cpu::restore_thread_affinity(prev); }
    } host_pin;
    if (glm_pool && glm_pool->ready()) {
        const std::vector<int> cores = strata::kernels::cpu::physical_cores(false);
        if (!cores.empty()) host_pin.prev = strata::kernels::cpu::pin_current_thread(cores[0]);
    }
    const auto t0 = Clock::now();
    for (int64_t q = P; q < n;) {
        int Tq = (int) std::min<int64_t>(T, n - q);
        if (state_at > q) Tq = (int) std::min<int64_t>(Tq, state_at - q);   // a window ends at the state dump
        for (int t = 0; t < Tq; ++t) {
            win[(size_t) t] = (int32_t) a.tokens[(size_t) (q + t)];
            nxt[(size_t) t] = q + t + 1 < n ? (int32_t) a.tokens[(size_t) (q + t + 1)] : -1;
        }
        if (x_file != nullptr &&
            std::fread(x_in.data(), sizeof(float), (size_t) Tq * g.hc * g.n_embd, x_file) != (size_t) Tq * g.hc * g.n_embd)
            return fail("reading the GLMT input");
        if (!eng.run(Tq, win.data(), q, err, x_file != nullptr ? x_in.data() : nullptr, from)) return fail(err);
        if (logpos != nullptr && !eng.window_logprobs(nxt.data(), Tq, q, extra, logpos, err)) return fail(err);
        for (int t = 0; logits_out && t < Tq; ++t)
            if (q + t >= logits_from && (!eng.copy_logits(t, lrow.data()) || !logits_out->append(lrow.data(), 1)))
                return fail("logits dump");
        if (full && a.stats) {
            int32_t am[8];
            if (!eng.argmax(Tq, am)) return fail("argmax");
            for (int t = 0; t < Tq; ++t)
                if (nxt[(size_t) t] >= 0) { agree += am[t] == nxt[(size_t) t]; ++rows; }
        }
        if (!eng.commit(Tq, err)) return fail(err);
        q += Tq;
        if (next_mark < marks.size() && q >= marks[next_mark]) {
            while (next_mark < marks.size() && q >= marks[next_mark]) ++next_mark;
            cudaDeviceSynchronize();
            const double ts = std::chrono::duration<double>(Clock::now() - t0).count();
            std::fprintf(stderr, "strata generate: glm5-next: mark %lld tokens %.3f s", (long long) q, ts);
            if (any_moe)
                std::fprintf(stderr, "; %s; file tier: %lld reads, %.1f MB, %.1f ms; RAM tier reads %lld",
                             experts.stats_line().c_str(), (long long) src.file_reads(),
                             (double) src.file_read_bytes() / 1e6, src.file_ms(), (long long) src.ram_reads());
            std::fprintf(stderr, "\n");
        }
        last_rows = Tq;
        if (P == 0 && !dump_state_at(q)) return fail(err);
    }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    // --max-new K: greedy continuation, one token per window, from the last row read
    if (a.max_new > 0) {
        if (!full) return fail("--max-new needs the whole model");
        std::vector<int32_t> gen;
        int32_t next = 0;
        {   // the last position read is the last row of the last window (or of the prompt path's head)
            int32_t am[8];
            if (!eng.argmax(last_rows, am)) return fail("argmax");
            next = am[last_rows - 1];
        }
        for (int64_t i = 0; i < a.max_new; ++i) {
            gen.push_back(next);
            if (i + 1 == a.max_new) break;
            const int64_t pos = n + i;
            if (!eng.run(1, &next, pos, err)) return fail(err);
            if (!eng.commit(1, err)) return fail(err);
            int32_t am[1];
            if (!eng.argmax(1, am)) return fail("argmax");
            next = am[0];
        }
        std::string line = "GLM_GEN";
        for (size_t i = 0; i < gen.size(); ++i) line += (i ? "," : " ") + std::to_string(gen[i]);
        std::printf("%s\n", line.c_str());
    }
    if (logpos != nullptr) std::fclose(logpos);
    for (auto& w : writers)
        if (w && !w->finish()) return fail("closing a dump file");
    if (logits_out && !logits_out->finish()) return fail("closing the logits dump");
    std::fprintf(stderr, "strata generate: glm5-next: %lld tokens through the windows in %.2f s: %.2f tok/s (windows of "
                         "%d%s)\n", (long long) (n - P), s, (double) (n - P) / std::max(s, 1e-9), T,
                 full ? "" : ", partial model");
    if (P > 0) {
        std::fprintf(stderr, "strata generate: %s\n", prefill.stats_line().c_str());
        if (const std::string r = prefill.timing_report(); !r.empty())
            std::fprintf(stderr, "strata generate: %s\n", r.c_str());
    }
    if (full && a.stats && rows > 0)
        std::fprintf(stderr, "strata generate: glm5-next: greedy pick = next prompt token at %lld of %lld positions\n",
                     (long long) agree, (long long) rows);
    if (any_moe) {
        std::fprintf(stderr, "strata generate: %s\n", experts.stats_line().c_str());
        std::fprintf(stderr, "strata generate: %s\n", experts.fetch_line().c_str());
        std::fprintf(stderr, "strata generate: glm5-next: file tier: %lld reads, %.1f MB, %.1f ms; RAM tier reads %lld\n",
                     (long long) src.file_reads(), (double) src.file_read_bytes() / 1e6, src.file_ms(),
                     (long long) src.ram_reads());
    }
    if (const std::string p = eng.profile_report(); !p.empty()) std::fprintf(stderr, "strata generate: %s\n", p.c_str());
    return 0;
}

}  // namespace strata::program
