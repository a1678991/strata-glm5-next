// src/program/glm_serve.cpp - `strata --serve` for GLM-5.3-Flash (glm5-next), port plan Stage 7's engine half.
//
// THE PROTOCOL is generate.cpp's serve loop's ("plan v0.3 P8: THE PERSISTENT ENGINE"), the subset serve/server.py
// drives for one request at a time, so the server runs unchanged:
//
//     -> INFO key=value ...            what the server's Monitor shows (context, kv, expert slots, conversation cache)
//     -> READY <max_context> stop      "stop": a STOP line ends the running request at its next step
//     <- GEN <max_new> [key=value ...] <id,id,...>
//     -> RESUME <n>                    n prompt tokens are not read again (the live session, a checkpoint, a parked
//                                      conversation in RAM or on disk)
//     -> PP <position> <prompt tokens> <ms> <tok/s>   after every prompt chunk
//     -> REUSED <n>                    the prompt is read; the first window comes next
//     -> T <id>                        each generated token, as soon as its window is done
//     -> DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused> <hits> <lookups>
//             <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
//     (ERR <message> when a request cannot run; STOP ends it; QUIT ends the process; GENI, BGEN and VRAM are refused:
//     no vision, no batch slots, no elastic cache on this path)
//
// Sampling keys as generate.cpp parses them (temperature, top_p, top_k, min_p, penalty_last_n, penalty_repeat,
// penalty_freq, penalty_present, seed); absent temperature = greedy.  The chain is the Qwen verifier's
// (kernels/sampler.hpp, sample_tokens on the head's logits, Philox(seed, position)); greedy takes the host argmax,
// as glm_generate's --max-new does.  No drafter: every decode window is one token.  A generated id in --eos-ids ends
// the reply ("stop"), --max-context bounds prompt + max_new as on the Qwen path.
//
// THE CONVERSATION CACHE (include/strata/core/glm_conversation.hpp for the state's halves).  `live` is what the
// session holds (the prompt read so far, then every committed token); `checks` is the chain of checkpoints, each a
// prefix of `live`; a request starts from the longest of: the live session, a checkpoint, a conversation parked in RAM
// (ConversationCache, --conversation-cache-mib/-slots) or on disk (ConversationDisk, --conversation-cache-disk*), and
// reads only the rest.  The outgoing conversation is parked before another one (or a checkpoint rewind) overwrites
// it.  Checkpoints are taken where the Qwen path takes them: in front of the prompt's last --turn-token (GLM:
// <|assistant|>, the new answer's turn), at the first turn boundary of a prompt read from token 0 when it lies at
// --prompt-cache-root or later (GLM: the first <|user|> / <|observation|> / turn token: the end of the system prompt
// and the tools), and every --prompt-cache-every fresh tokens; --prompt-cache N keeps at most N (the root pinned,
// conv_cache.hpp).
//
// THE PROMPT is read in parts: [resume, root) [root, turn) [turn, n - 1), each either through the prompt path in
// chunks (GlmPrefill, a PP line after each) or, when it has at most --short-read tokens, through teacher-forced windows
// of --spec T tokens (default 1); the last prompt token goes through the first decode window.  The prompt path reads
// any position (pos0 > 0: the multi-turn continuation), with the session holding exactly the cells before it.
//
// Debugging: STRATA_GLM_LOGITS_HASH=1 adds a hash of every decode row's full logits to the request's log line (the
// byte-identity check of a restored conversation against an uninterrupted one).
#include "strata/program/glm_serve.hpp"

#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/glm.hpp"
#include "strata/core/glm_conversation.hpp"
#include "strata/core/glm_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/prefill/glm_prefill.hpp"
#include "strata/program/conv_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::program {
namespace {

using Clock = std::chrono::steady_clock;
namespace sc = strata::core;
namespace sk = strata::kernels;
using ConvCheckpoint = sc::ConversationCheckpoint;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// "a,b,c" -> ids; false on anything else
bool parse_ids(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    while (*s == ' ') ++s;
    if (*s == '\0') { err = "no token ids"; return false; }
    while (*s != '\0') {
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || errno != 0) { err = "bad token id list"; return false; }
        out.push_back(v);
        s = end;
        while (*s == ' ') ++s;
        if (*s == ',') ++s;
        else if (*s != '\0') { err = "bad token id list"; return false; }
    }
    return true;
}

// the disk tier's fingerprint parts (generate.cpp's file_id)
std::string file_id(const std::string& path, bool content) {
    std::string id = path;
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    id += ec ? std::string(":missing") : ":" + std::to_string((unsigned long long) size);
    if (content && !ec) {
        uint64_t h = 0xcbf29ce484222325ull;
        if (FILE* f = std::fopen(path.c_str(), "rb")) {
            for (int c; (c = std::fgetc(f)) != EOF;) { h ^= (uint8_t) c; h *= 0x100000001b3ull; }
            std::fclose(f);
        }
        char hex[24];
        std::snprintf(hex, sizeof(hex), ":%016llx", (unsigned long long) h);
        id += hex;
    }
    return id;
}

}  // namespace

int glm_serve(const GlmServeParts& P) {
    const GlmRunArgs& a = *P.args;
    const sc::ModelGeometry& g = *P.g;
    sc::GlmEngine& eng = *P.eng;
    strata::prefill::GlmPrefill* pf = P.prefill;
    const sc::GlmSessionView view = eng.session_view();
    const int W = std::max(1, std::min(eng.config().max_t, P.window));
    const int64_t max_ctx = eng.config().max_cells;
    const int64_t chunk = pf != nullptr ? pf->chunk() : 0;
    std::string err;
    if (a.tail_role_token >= 0)
        std::fprintf(stderr, "strata serve: glm5-next: --tail-role-token is not used on this path (GLM's turns open "
                             "with their role token)\n");
    if (!a.expert_profile_save.empty())
        std::fprintf(stderr, "strata serve: glm5-next: --expert-profile-save: nothing to save, this path has no "
                             "adaptive expert tier\n");
    std::vector<int64_t> eos = a.eos_ids;
    // the root boundary (see the header): the end ids that open a turn (<|user|>, <|observation|>) and the turn token
    auto root_token = [&](int64_t id) {
        return id == a.turn_token || std::find(eos.begin(), eos.end(), id) != eos.end();
    };

    // ---- the conversation cache
    std::vector<int32_t> live;
    bool live_ok = false;
    std::vector<ConvCheckpoint> checks;
    uint64_t check_clock = 0;
    const std::vector<sc::ConversationImageKey> no_imgs;
    sc::ConversationCache conversations(a.prompt_cache > 0 ? (size_t) a.conversation_cache_mib * 1024 * 1024 : 0,
                                        (size_t) a.conversation_cache_slots);
    if (a.conversation_cache_mib > 0 && (a.prompt_cache == 0 || a.conversation_cache_slots == 0))
        std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                     a.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    sc::ConversationDisk disk;
    if (!a.conversation_cache_disk.empty()) {
        if (!conversations.enabled()) {
            std::fprintf(stderr, "strata serve: conversation cache disk: off (needs --conversation-cache-mib > 0)\n");
        } else {
            // everything that changes what a saved state means; "arch=glm5-next" keeps GLM's directory apart from any
            // Qwen one (the geometry key is tagged too).  The DSA rows are saved in the identity layout, so
            // --kv-resident does not change an image and is not part of it.
            std::string fp = std::string("engine=") + STRATA_VERSION + " arch=glm5-next";
            fp += " pack=" + file_id(a.pack + "/index.txt", true) + "," + file_id(a.pack + "/native_experts.txt", true);
            for (const auto& shard : P.shards) fp += " native=" + file_id(shard, false);
            fp += " kv=" + std::string(eng.config().kv_fmt == sk::kDsaLatF16 ? "f16" : "int8") +
                  " max_context=" + std::to_string(max_ctx);
            std::string derr;
            disk.set_keep_superseded(a.conversation_cache_disk_keep_superseded);
            if (!disk.open(a.conversation_cache_disk, fp, (size_t) a.conversation_cache_disk_mib * 1024 * 1024, derr))
                std::fprintf(stderr, "strata serve: conversation cache disk: off (%s)\n", derr.c_str());
            else
                std::fprintf(stderr, "strata serve: conversation cache disk: %s, %zu entries, %zu MiB (limit %lld MiB); "
                                     "superseded turns %s\n", disk.dir().c_str(), disk.size(), disk.bytes() >> 20,
                             (long long) a.conversation_cache_disk_mib, disk.keep_superseded() ? "kept" : "deleted");
        }
    }
    auto reap_disk = [&](bool wait, size_t held) {
        if (auto img = disk.finish(wait)) {
            const size_t n = img->live.ids.size();
            const bool stored = conversations.put(std::move(*img), held);
            std::fprintf(stderr, "strata serve: conversation cache: %s %zu tokens after their disk write; "
                                 "parked=%zu bytes=%zu\n", stored ? "parked" : "skipped", n, conversations.size(),
                         conversations.bytes());
        }
    };
    // the outgoing conversation into the RAM tier (and through to disk); false only when a copy itself failed
    auto park_current = [&](size_t held) -> bool {
        if (!conversations.enabled() || !live_ok || live.empty()) return true;
        if (disk.busy()) reap_disk(true, held);
        if (const size_t dropped = conversations.drop_superseded(live, no_imgs, checks, true))
            std::fprintf(stderr, "strata serve: conversation cache: dropped %zu superseded cop%s of this "
                                 "conversation; parked=%zu\n", dropped, dropped == 1 ? "y" : "ies", conversations.size());
        const sc::ConversationView cv{live, no_imgs, checks, true};
        size_t estimate = 0;
        std::string e;
        if (!sc::glm_snapshot_bytes(cv, view, estimate, e)) {
            std::fprintf(stderr, "strata serve: conversation cache: skip parking (%s)\n", e.c_str());
            return true;
        }
        if (!conversations.make_room(estimate, held)) {
            std::fprintf(stderr, "strata serve: conversation cache: skip parking (snapshot %zu MiB exceeds available "
                                 "budget)\n", estimate >> 20);
            return true;
        }
        const auto t0 = Clock::now();
        try {
            const uint64_t floor = (uint64_t) a.conversation_cache_min_free_mib * 1024 * 1024;
            if (!sc::conversation_memory_admit(sc::conversation_available_memory(), estimate, floor)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM admission; need %zu "
                                     "MiB plus %lld MiB floor, or telemetry unavailable)\n", estimate >> 20,
                             (long long) a.conversation_cache_min_free_mib);
                return true;
            }
            sc::SavedConversation image;
            if (!sc::glm_snapshot_save(image, cv, view, e)) {
                err = e;
                return false;
            }
            const size_t snapshot_bytes = image.bytes();
            bool writing = false;
            if (disk.enabled()) {
                std::string derr;
                switch (disk.begin_store(std::move(image), derr)) {
                    case sc::ConversationDisk::Store::started: writing = true; break;
                    case sc::ConversationDisk::Store::unchanged:
                        std::fprintf(stderr, "strata serve: conversation cache disk: unchanged on disk\n");
                        break;
                    case sc::ConversationDisk::Store::refused:
                        std::fprintf(stderr, "strata serve: conversation cache disk: not written (%s)\n", derr.c_str());
                        break;
                }
            }
            const bool stored = writing || conversations.put(std::move(image), held);
            std::fprintf(stderr, "strata serve: conversation cache: %s %zu tokens in %.1f ms%s; parked=%zu bytes=%zu "
                                 "evictions=%zu snapshot_bytes=%zu reused_kv_bytes=0\n",
                         stored ? "parked" : "skipped", live.size(), ms_since(t0),
                         writing ? " (writing it to disk in the background, then to RAM)" : "", conversations.size(),
                         conversations.bytes(), conversations.evictions(), snapshot_bytes);
        } catch (const std::bad_alloc&) {
            std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking\n");
        }
        return true;
    };

    // ---- stdin on its own thread, so a STOP reaches a running request (generate.cpp's reader: read(2), not cin)
    std::atomic<bool> stop_req{false};
    std::mutex in_mu;
    std::condition_variable in_cv;
    std::deque<std::string> in_lines;
    bool in_eof = false;
    std::thread([&] {
        std::string l, buf;
        char chunk_buf[4096];
        auto getline_fd = [&](std::string& out) -> bool {
            for (;;) {
                const size_t nl = buf.find('\n');
                if (nl != std::string::npos) {
                    out.assign(buf, 0, nl);
                    buf.erase(0, nl + 1);
                    return true;
                }
#if defined(_WIN32)
                const int n = _read(0, chunk_buf, (unsigned) sizeof chunk_buf);
#else
                const ssize_t n = ::read(0, chunk_buf, sizeof chunk_buf);
                if (n < 0 && errno == EINTR) continue;
#endif
                if (n <= 0) {
                    if (buf.empty()) return false;
                    out.swap(buf);
                    buf.clear();
                    return true;
                }
                buf.append(chunk_buf, (size_t) n);
            }
        };
        while (getline_fd(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop_req.store(true); continue; }
            std::lock_guard<std::mutex> lk(in_mu);
            in_lines.push_back(l);
            in_cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(in_mu);
        in_eof = true;
        in_cv.notify_one();
    }).detach();
    auto next_line = [&](std::string& out) -> bool {
        std::unique_lock<std::mutex> lk(in_mu);
        in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
        if (in_lines.empty()) return false;
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };

    // ---- what the server's Monitor shows, then READY
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t slots = P.cache != nullptr ? (int64_t) P.cache->resident() : 0;
        const int64_t mib = P.cache != nullptr ? (int64_t) (P.cache->gib() * 1024.0) : 0;
        std::fprintf(stderr, "strata serve: glm5-next: %lld MiB of VRAM free with everything loaded; context %lld, "
                             "DSA latent %s (%s), session %.0f MiB of VRAM + %.0f MiB page-locked; prompt path %s; "
                             "decode windows of 1, teacher-forced reads in windows of %d\n",
                     (long long) (free_b >> 20), (long long) max_ctx,
                     eng.config().kv_fmt == sk::kDsaLatF16 ? "f16" : "int8",
                     sc::glm_kv_slots(g, eng.config()) > 0
                         ? ("streamed, " + std::to_string((long long) eng.config().kv_resident) + " cells per layer "
                            "in VRAM").c_str()
                         : "fully resident",
                     (double) eng.session_bytes() / 1048576.0, (double) eng.session_host_bytes() / 1048576.0,
                     pf != nullptr ? ("chunks of " + std::to_string((long long) chunk) + " tokens").c_str()
                                   : "off (every prompt token through the windows)",
                     W);
        std::printf("INFO context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                    "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=0 mtp_max=0 lookup=0 "
                    "vram_free_mib=%lld cvec=0 arena_mib=%lld pool_workers=%d pcie_frac=0.00 spec_min_p=0.00 "
                    "conversation_cache_mib=%lld conversation_cache_slots=%d conversation_cache_min_free_mib=%lld "
                    "tail_role_token=%lld vram_elastic=0 arch=glm5-next prompt_chunk=%lld engine=" STRATA_VERSION "\n",
                    (long long) max_ctx, eng.config().kv_fmt == sk::kDsaLatF16 ? "f16" : "int8",
                    (long long) (sc::glm_kv_slots(g, eng.config()) > 0 ? eng.config().kv_resident : 0),
                    (long long) slots, (long long) mib, (long long) slots, (long long) mib, (long long) (free_b >> 20),
                    (long long) (P.src != nullptr ? (P.src->resident_bytes() >> 20) : 0), P.pool_workers,
                    (long long) a.conversation_cache_mib, a.conversation_cache_slots,
                    (long long) a.conversation_cache_min_free_mib, (long long) a.tail_role_token, (long long) chunk);
        std::printf("READY %lld stop\n", (long long) max_ctx);
        std::fflush(stdout);
    }

    static const bool logits_hash = std::getenv("STRATA_GLM_LOGITS_HASH") != nullptr;
    std::vector<float> lrow(logits_hash ? (size_t) g.n_vocab : 0);
    std::string line;
    int64_t requests = 0;
    while (next_line(line)) {
        if (line == "QUIT") break;
        stop_req.store(false);   // a STOP that arrived between requests is stale
        err.clear();
        if (line == "VRAM" || line.rfind("VRAM ", 0) == 0) {
            std::printf("ERR VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)\n");
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GENI ", 0) == 0) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> <id,id,...> (glm5-next: no images, no batch slots)\n");
            std::fflush(stdout);
            continue;
        }
        // ---- the request line: GEN <max_new> [key=value ...] <ids>
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        float req_temperature = 0.0f, req_top_p = 1.0f, req_min_p = 0.0f;
        int req_top_k = 20;   // the sampler's own default; the sampled path keeps 1..64
        unsigned long long req_seed = 0;
        float req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
        int req_penalty_last_n = 0;
        if (endp != nullptr) {
            for (;;) {
                while (*endp == ' ') ++endp;
                const char* start = endp;
                while (*endp != '\0' && *endp != ' ') ++endp;
                if (endp == start) break;
                const std::string tok(start, (size_t) (endp - start));
                const size_t eq = tok.find('=');
                if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
                const std::string key = tok.substr(0, eq);
                const char* val = tok.c_str() + eq + 1;
                const float fv = std::strtof(val, nullptr);
                if (key == "temperature") req_temperature = fv;
                else if (key == "top_p") req_top_p = fv;
                else if (key == "top_k") req_top_k = std::atoi(val);
                else if (key == "min_p") req_min_p = fv;
                else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(val);
                else if (key == "penalty_repeat") req_penalty_repeat = fv;
                else if (key == "penalty_freq") req_penalty_freq = fv;
                else if (key == "penalty_present") req_penalty_present = fv;
                else if (key == "seed") req_seed = std::strtoull(val, nullptr, 10);
                // unknown keys (cvec, pcie_frac, spec_min_p: Qwen-only) are skipped
            }
        }
        std::vector<int64_t> ids64;
        std::string pe;
        if (max_new < 1 || endp == nullptr || !parse_ids(endp, ids64, pe)) {
            std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
            std::fflush(stdout);
            continue;
        }
        const int64_t n = (int64_t) ids64.size();
        if (n + max_new + 8 > max_ctx) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                        (long long) max_new, (long long) max_ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int64_t t : ids64) bad = bad || t < 0 || t >= g.n_vocab;
        if (bad) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        std::vector<int32_t> ids(ids64.begin(), ids64.end());
        ++requests;
        const Clock::time_point r0 = Clock::now();
        auto starts_with = [&](const std::vector<int32_t>& pre) -> bool {
            const int64_t L = (int64_t) pre.size();
            return L >= 1 && L <= n - 1 && std::equal(pre.begin(), pre.end(), ids.begin());
        };
        // ---- where this request starts reading
        int64_t resume = 0;
        bool from_live = false;
        if (a.prompt_cache > 0) {
            if (live_ok && starts_with(live)) { resume = (int64_t) live.size(); from_live = true; }
            for (const ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() > resume && starts_with(c.ids)) {
                    resume = (int64_t) c.ids.size();
                    from_live = false;
                }
        }
        if (disk.busy()) {
            const int64_t pending = disk.in_flight_match(ids, no_imgs, true);
            reap_disk(pending > resume, 0);
        }
        auto parked = conversations.best(ids, no_imgs, true);
        std::optional<sc::SavedConversation> incoming;
        if (disk.enabled()) {
            const auto dm = disk.best(ids, no_imgs, true);
            if (dm.tokens > std::max(resume, parked.tokens)) {
                const uint64_t floor = (uint64_t) a.conversation_cache_min_free_mib * 1024 * 1024;
                if (!sc::conversation_memory_admit(sc::conversation_available_memory(), dm.size, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache disk: skip restore (physical RAM admission; "
                                         "need %zu MiB plus %lld MiB floor)\n", dm.size >> 20,
                                 (long long) a.conversation_cache_min_free_mib);
                } else {
                    const auto td = Clock::now();
                    sc::SavedConversation image;
                    std::string derr;
                    try {
                        if (disk.load(dm, ids, no_imgs, image, derr)) {
                            incoming.emplace(std::move(image));
                            parked.tokens = dm.tokens;
                            parked.live = dm.live;
                            std::fprintf(stderr, "strata serve: conversation cache disk: read %lld tokens (%s) in "
                                                 "%.1f ms\n", (long long) dm.tokens, dm.live ? "live" : "checkpoint",
                                         ms_since(td));
                        } else {
                            std::fprintf(stderr, "strata serve: conversation cache disk: not restored (%s)\n",
                                         derr.c_str());
                        }
                    } catch (const std::bad_alloc&) {
                        std::fprintf(stderr, "strata serve: conversation cache disk: allocation failed; skip restore\n");
                    }
                }
            }
        }
        if (!incoming && parked.tokens > resume) incoming.emplace(conversations.take(parked.index));
        if (incoming && !sc::glm_snapshot_validate(*incoming, view, err)) {
            std::fprintf(stderr, "strata serve: conversation cache: discard invalid snapshot (%s)\n", err.c_str());
            incoming.reset();
            err.clear();
        }
        // the outgoing branch is parked before a rewind, a reset or an incoming image overwrites its cells
        if ((!from_live || incoming) && !park_current(incoming ? incoming->bytes() : 0)) {
            std::printf("ERR parking the conversation failed: %s\n", err.c_str());
            std::fflush(stdout);
            return 1;
        }
        if (incoming) {
            const auto t0 = Clock::now();
            if (sc::glm_snapshot_restore(*incoming, view, err) != sc::ConversationRestore::restored) {
                // validated above: a failure here may leave the session half written - never decode from it
                std::printf("ERR restoring parked conversation: %s\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
            live = std::move(incoming->live.ids);
            checks = std::move(incoming->checkpoints);
            resume = parked.tokens;
            from_live = parked.live;
            incoming.reset();
            std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu "
                                 "bytes=%zu\n", (long long) resume, from_live ? "live" : "checkpoint", ms_since(t0),
                         conversations.size(), conversations.bytes());
        }
        // this request rewrites every cell from `resume` on: the checkpoints kept are prefixes of both
        checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& c) {
                         return (int64_t) c.ids.size() > resume || !starts_with(c.ids);
                     }), checks.end());
        live_ok = false;   // until this request has finished, the session is in between
        if (resume == 0) {
            if (!eng.reset(err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
            checks.clear();
        } else if (!from_live) {
            ConvCheckpoint* c = nullptr;
            for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
            if (c != nullptr) c->used = ++check_clock;
            if (c == nullptr || !sc::glm_checkpoint_restore(*c, view, err)) {
                std::printf("ERR restoring a conversation checkpoint failed%s%s\n", err.empty() ? "" : ": ",
                            err.c_str());
                std::fflush(stdout);
                return 1;
            }
        }
        const int64_t read_from = resume;
        std::printf("RESUME %lld\n", (long long) resume);
        std::fflush(stdout);

        // ---- the prompt: [resume, root) [root, turn) [turn, n - 1), checkpoints at root and turn
        auto checkpoint_at = [&](int64_t L) -> bool {
            if (a.prompt_cache <= 0 || L < 1) return true;
            for (ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() == L) { c.used = ++check_clock; return true; }
            ConvCheckpoint c;
            c.ids.assign(ids.begin(), ids.begin() + L);
            if (!sc::glm_checkpoint_save(c, view, err)) return false;
            c.used = ++check_clock;
            checks.push_back(std::move(c));
            while ((int) checks.size() > a.prompt_cache) {
                std::vector<uint64_t> stamps;
                for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
                checks.erase(checks.begin() + (std::ptrdiff_t) conv_cache::eviction_victim(stamps.data(), stamps.size(),
                                                                                           a.prompt_cache));
            }
            return true;
        };
        int64_t turn_at = -1, root_at = -1;
        if (a.prompt_cache > 0 && a.turn_token >= 0)
            for (int64_t i = n - 1; i > resume; --i)
                if (ids[(size_t) i] == a.turn_token) { turn_at = i; break; }
        if (a.prompt_cache > 0 && a.prompt_cache_root > 0 && read_from == 0 && turn_at > 0)
            for (int64_t i = 1; i < turn_at; ++i)
                if (root_token(ids[(size_t) i])) {
                    if (i >= a.prompt_cache_root) root_at = i;
                    break;
                }
        bool cancelled = false;
        int64_t pp_reached = read_from;
        int64_t next_check = a.prompt_cache_every > 0 ? read_from + a.prompt_cache_every : INT64_MAX;
        // [x, y) through teacher-forced windows of W tokens
        auto read_windows = [&](int64_t x, int64_t y) -> bool {
            for (int64_t q = x; q < y;) {
                const int Tq = (int) std::min<int64_t>(W, y - q);
                if (!eng.run(Tq, ids.data() + q, q, err) || !eng.commit(Tq, err)) return false;
                q += Tq;
                pp_reached = q;
                if (stop_req.load()) { cancelled = true; return true; }
            }
            return true;
        };
        // [x, y) through the prompt path, one chunk per call, a PP line after each
        auto read_prompt = [&](int64_t x, int64_t y) -> bool {
            for (int64_t q = x; q < y;) {
                const int64_t m = std::min<int64_t>(chunk, y - q);
                if (!pf->run(ids.data() + q, m, q, 0, err)) return false;
                q += m;
                pp_reached = q;
                const double ms = ms_since(r0);
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) q, (long long) n, ms,
                            ms > 0.0 ? 1000.0 * (double) (q - read_from) / ms : 0.0);
                std::fflush(stdout);
                if (q >= next_check && q < y) {   // a mid-prompt checkpoint (the part's end takes its own below)
                    if (!checkpoint_at(q)) return false;
                    next_check = q + a.prompt_cache_every;
                }
                if (stop_req.load()) { cancelled = true; return true; }
            }
            return true;
        };
        int64_t at = read_from;
        for (const int64_t to : {root_at, turn_at, n - 1}) {
            if (to <= at || cancelled) continue;
            const bool win = pf == nullptr || to - at <= a.short_read;
            if (!(win ? read_windows(at, to) : read_prompt(at, to))) {
                std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
            if (cancelled) break;
            at = to;
            if ((to == turn_at || to == root_at) && !checkpoint_at(to)) {
                std::printf("ERR saving a conversation checkpoint failed: %s\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
        }
        const double prompt_ms = ms_since(r0);
        if (!cancelled) {
            std::printf("REUSED %lld\n", (long long) resume);
            std::fflush(stdout);
        }

        // ---- decode: one token per window, the first holding the last prompt token
        sk::SamplerParams req_sp;
        req_sp.greedy = req_temperature <= 0.0f;
        req_sp.temperature = req_temperature;
        req_sp.top_p = req_top_p;
        req_sp.top_k = req_top_k;
        req_sp.seed = req_seed ? req_seed : (unsigned long long) Clock::now().time_since_epoch().count();
        req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
        req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
        req_sp.penalty_repeat = req_penalty_repeat;
        req_sp.penalty_freq = req_penalty_freq;
        req_sp.penalty_present = req_penalty_present;
        const int hist_n = std::min(req_sp.penalty_last_n, sc::GlmEngine::kSampleHistoryCap);
        const bool sampled = !req_sp.greedy || hist_n > 0;
        std::vector<int32_t> hist((size_t) std::max(hist_n, 1));
        std::vector<int32_t> consumed;
        consumed.reserve((size_t) (n + max_new + 1));
        for (int64_t i = 0; i < n - 1; ++i) consumed.push_back(ids[(size_t) i]);
        const char* finish = cancelled ? "cancel" : "length";
        const sc::GlmExperts::Stats st0 = P.experts != nullptr ? P.experts->stats() : sc::GlmExperts::Stats{};
        const int64_t file0 = P.src != nullptr ? P.src->file_reads() : 0;
        const double file_mb0 = P.src != nullptr ? (double) P.src->file_read_bytes() / 1e6 : 0.0;
        int64_t produced = 0;
        uint64_t lhash = 1469598103934665603ull;
        const Clock::time_point d0 = Clock::now();
        int64_t p = n - 1;
        int32_t x = ids[(size_t) (n - 1)];
        while (!cancelled && produced < max_new) {
            if (p + 1 > max_ctx) break;   // "length"
            if (!eng.run(1, &x, p, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fflush(stdout);
                return 1;
            }
            if (logits_hash) {
                if (!eng.copy_logits(0, lrow.data())) { std::printf("ERR logits\n"); return 1; }
                lhash = fnv1a(lrow.data(), lrow.size() * sizeof(float), lhash);
            }
            int32_t y = 0;
            bool ok = true;
            if (!sampled) {
                ok = eng.argmax(1, &y);
            } else {
                if (hist_n > 0) sk::penalty_rows(consumed.data(), (int64_t) consumed.size(), &x, 1, hist_n, hist.data());
                ok = eng.sample(1, req_sp, hist_n > 0 ? hist.data() : nullptr, hist_n, &y, err);
            }
            if (!ok || !eng.commit(1, err)) {
                std::printf("ERR %s\n", err.empty() ? "picking the next token failed" : err.c_str());
                std::fflush(stdout);
                return 1;
            }
            consumed.push_back(x);
            std::printf("T %d\n", (int) y);
            std::fflush(stdout);
            ++produced;
            if (std::find(eos.begin(), eos.end(), (int64_t) y) != eos.end()) { finish = "stop"; break; }
            if (stop_req.load()) { finish = "cancel"; break; }
            x = y;
            ++p;
        }
        const double decode_ms = ms_since(d0);
        if (!cancelled) {
            // a prompt stopped halfway leaves the session between two parts: nothing to continue from (the
            // checkpoints taken while reading it are still good)
            live.swap(consumed);
            live_ok = a.prompt_cache > 0;
        }
        const sc::GlmExperts::Stats st1 = P.experts != nullptr ? P.experts->stats() : sc::GlmExperts::Stats{};
        const int64_t hits = st1.vram - st0.vram;
        const int64_t cpu = st1.cpu - st0.cpu;
        const int64_t ram = st1.ram - st0.ram, file = st1.file - st0.file;
        const int64_t fresh = n - resume;
        const int64_t read_n = cancelled ? std::clamp<int64_t>(pp_reached - resume, 0, fresh) : fresh;
        std::printf("DONE %lld %lld %.1f %.1f %s 0 0 %lld %lld %lld %lld %lld %.1f %lld %lld\n", (long long) produced,
                    (long long) n, prompt_ms, decode_ms, finish, (long long) resume, (long long) hits,
                    (long long) (hits + cpu), (long long) ram,
                    (long long) (P.src != nullptr ? P.src->file_reads() - file0 : 0),
                    (P.src != nullptr ? (double) P.src->file_read_bytes() / 1e6 : 0.0) - file_mb0, (long long) read_n,
                    (long long) (ram - cpu + file));
        std::fflush(stdout);
        char read_txt[64];
        if (cancelled)
            std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
        else
            std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
        char hash_txt[48] = "";
        if (logits_hash) std::snprintf(hash_txt, sizeof hash_txt, "; logits %016llx", (unsigned long long) lhash);
        std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), %lld "
                             "generated in %.0f ms (%.1f tok/s), drafts accepted 0 of 0, %zu checkpoints%s%s\n",
                     (long long) n, (long long) resume, read_txt, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double) read_n / prompt_ms : 0.0, (long long) produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, checks.size(),
                     cancelled ? " (cancelled)" : "", hash_txt);
        if (P.experts != nullptr)
            std::fprintf(stderr, "strata serve: glm experts this request: %lld from VRAM, %lld from RAM (%lld on the CPU "
                                 "pool), %lld from the file tier\n", (long long) hits, (long long) ram, (long long) cpu,
                         (long long) file);
    }
    // ---- the live conversation to disk at exit, so it survives a restart
    if (disk.busy()) reap_disk(true, 0);
    if (disk.enabled() && live_ok && !live.empty()) {
        const auto td = Clock::now();
        try {
            const sc::ConversationView cv{live, no_imgs, checks, true};
            sc::SavedConversation image;
            std::string derr;
            if (sc::glm_snapshot_save(image, cv, view, derr) && disk.store(image, derr))
                std::fprintf(stderr, "strata serve: conversation cache disk: wrote the live conversation (%zu tokens) "
                                     "at exit in %.1f ms\n", live.size(), ms_since(td));
            else
                std::fprintf(stderr, "strata serve: conversation cache disk: live conversation not written (%s)\n",
                             derr.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata serve: conversation cache disk: live conversation not written (%s)\n", e.what());
        }
    }
    std::fprintf(stderr, "strata serve: glm5-next: %lld requests served\n", (long long) requests);
    return 0;
}

}  // namespace strata::program
