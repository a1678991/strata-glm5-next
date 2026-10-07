// tests/glm/glm_conversation_test.cpp - --serve's conversation cache for glm5-next on synthetic device state
// (include/strata/core/glm_conversation.hpp).  No model: a session view over device buffers of a small glm5-next
// geometry (4 KDA layers, 2 DSA layers), filled with random bytes, so every byte that is saved, parked, written to disk
// and restored is checked exactly.
//
//   1. a checkpoint (the KDA running state) restores bitwise after the state was overwritten;
//   2. a parked image (live + checkpoints + every DSA layer's cells [0, L) and pooled rows) restores bitwise into a
//      scrambled session, fully resident and streamed (--kv-resident: the host copy is authoritative), and across the
//      two residencies;
//   3. the RAM tier (ConversationCache) holds it within the estimate glm_snapshot_bytes gives (put()'s accounting);
//   4. the disk tier (ConversationDisk, unchanged) round-trips it: written, the directory reopened, matched by a
//      prompt that extends it, read back, validated and restored bitwise - also with keep-superseded;
//   5. validation refuses a Qwen-keyed image, another DSA format, a wrong length, a missing layer, a checkpoint that
//      is not a prefix of the live tokens; the Qwen and GLM keys differ.
// Exit 77 (skipped) without a GPU.
#include "strata/core/conversation_disk.hpp"
#include "strata/core/glm_conversation.hpp"
#include "strata/kernels/dsa.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace sc = strata::core;
namespace sk = strata::kernels;
namespace fs = std::filesystem;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}
#define CK(call)                                                                                                     \
    do {                                                                                                             \
        const cudaError_t e_ = (call);                                                                               \
        if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); std::exit(2); }   \
    } while (0)

sc::ModelGeometry geometry() {
    sc::ModelGeometry g;
    g.arch = sc::GeometryArch::Glm5Next;
    g.n_embd = 256;
    g.n_layers = 6;
    g.attn_layer_mask = (1ull << 2) | (1ull << 5);   // layers 2 and 5 are DSA, the rest KDA
    g.n_dense_lead = 1;
    g.ssm_state_size = 128;
    g.ssm_v_heads = 2;
    g.ssm_k_heads = 2;
    g.ssm_d_conv = 4;
    g.ssm_qk_dim = 256;
    g.ssm_conv_separate = true;
    g.kv_lora = 512;
    g.idx_key_dim = 128;
    g.idx_q_heads = 4;
    g.idx_kpool = 4;
    g.idx_top_k = 2048;
    g.n_head = 4;
    g.n_vocab = 1000;
    g.n_expert = 8;
    g.n_ff = 64;
    return g;
}

sk::DsaShapes dsa_shapes(const sc::ModelGeometry& g) {
    sk::DsaShapes s;
    s.n_embd = g.n_embd; s.n_head = g.n_head; s.q_lora = 64; s.kv_lora = g.kv_lora; s.head_k = 64; s.head_v = 64;
    s.idx_n_head = g.idx_q_heads; s.idx_dim = g.idx_key_dim; s.kpool = g.idx_kpool; s.idx_top_k = g.idx_top_k;
    return s;
}

// one synthetic session: device buffers for every layer, a stream; `slots` 0 = fully resident
struct Session {
    sc::ModelGeometry g;
    sc::GlmConfig cfg;
    sc::GlmSessionView v;
    std::vector<void*> dev;
    std::vector<void*> host;
    cudaStream_t st = nullptr;
    size_t hist = 0, state = 0;

    Session(int64_t max_cells, int64_t kv_resident) {
        g = geometry();
        cfg.max_t = 1;
        cfg.max_cells = max_cells;
        cfg.kv_resident = kv_resident;
        CK(cudaStreamCreate(&st));
        v.g = &g;
        v.cfg = &cfg;
        v.dshape = dsa_shapes(g);
        v.stream = st;
        v.n_run = g.n_layers;
        v.kda_hist.assign((size_t) g.n_layers, nullptr);
        v.kda_state.assign((size_t) g.n_layers, nullptr);
        v.dsa.assign((size_t) g.n_layers, sk::DsaKvCache{});
        hist = (size_t) 3 * g.ssm_qk_dim * (g.ssm_d_conv - 1) * 4;
        state = (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4;
        const int64_t pages = (max_cells + 3) / 4;
        const int64_t slots = kv_resident > 0 && (kv_resident + 3) / 4 < pages ? (kv_resident + 3) / 4 : 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (g.attn_layer(l)) {
                void* p = nullptr;
                CK(cudaMalloc(&p, sk::dsa_kv_bytes(v.dshape, cfg.kv_fmt, max_cells, slots)));
                dev.push_back(p);
                sk::dsa_kv_carve(p, v.dshape, cfg.kv_fmt, max_cells, slots, v.dsa[(size_t) l]);
                if (slots > 0) {
                    void* h = nullptr;
                    CK(cudaHostAlloc(&h, sk::dsa_kv_host_bytes(v.dshape, cfg.kv_fmt, max_cells), cudaHostAllocMapped));
                    host.push_back(h);
                    void* d = nullptr;
                    CK(cudaHostGetDevicePointer(&d, h, 0));
                    sk::dsa_kv_host_carve(d, v.dshape, cfg.kv_fmt, max_cells, v.dsa[(size_t) l]);
                }
                sk::dsa_kv_reset(v.dsa[(size_t) l], v.dshape, st);
            } else {
                void* a = nullptr;
                void* b = nullptr;
                CK(cudaMalloc(&a, hist));
                CK(cudaMalloc(&b, state));
                dev.push_back(a);
                dev.push_back(b);
                v.kda_hist[(size_t) l] = (float*) a;
                v.kda_state[(size_t) l] = (float*) b;
            }
        }
        CK(cudaStreamSynchronize(st));
    }
    ~Session() {
        cudaDeviceSynchronize();
        for (void* p : dev) cudaFree(p);
        for (void* p : host) cudaFreeHost(p);
        cudaStreamDestroy(st);
    }
    bool streamed(int64_t l) const { return v.dsa[(size_t) l].n_slots < v.dsa[(size_t) l].n_pages; }
    // the identity rows of a DSA layer (the host copy when streamed)
    void* lat(int64_t l) const { const auto& c = v.dsa[(size_t) l]; return streamed(l) ? (void*) c.host.lat_q : (void*) c.lat_q; }
    void* scl(int64_t l) const { const auto& c = v.dsa[(size_t) l]; return streamed(l) ? (void*) c.host.lat_scale : (void*) c.lat_scale; }
    void* kg(int64_t l) const { const auto& c = v.dsa[(size_t) l]; return streamed(l) ? (void*) c.host.kg : (void*) c.kg; }

    static void fill(void* p, size_t n, std::mt19937& rng) {
        std::vector<uint8_t> b(n);
        for (auto& x : b) x = (uint8_t) rng();
        CK(cudaMemcpy(p, b.data(), n, cudaMemcpyDefault));
    }
    void scramble_kda(std::mt19937& rng) {
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (!g.attn_layer(l)) { fill(v.kda_hist[(size_t) l], hist, rng); fill(v.kda_state[(size_t) l], state, rng); }
    }
    void scramble(std::mt19937& rng) {
        scramble_kda(rng);
        const int64_t cells = cfg.max_cells;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!g.attn_layer(l)) continue;
            fill(lat(l), (size_t) cells * 512, rng);
            fill(scl(l), (size_t) cells * 16, rng);
            fill(kg(l), (size_t) cells * 512, rng);
            fill(v.dsa[(size_t) l].pooled, (size_t) v.dsa[(size_t) l].max_pools * 256, rng);
        }
    }
    static uint64_t hash(const void* p, size_t n, uint64_t h) {
        std::vector<uint8_t> b(n);
        CK(cudaMemcpy(b.data(), p, n, cudaMemcpyDefault));
        for (uint8_t x : b) { h ^= x; h *= 1099511628211ull; }
        return h;
    }
    uint64_t kda_hash() const {
        uint64_t h = 1469598103934665603ull;
        for (int64_t l = 0; l < g.n_layers; ++l)
            if (!g.attn_layer(l)) { h = hash(v.kda_hist[(size_t) l], hist, h); h = hash(v.kda_state[(size_t) l], state, h); }
        return h;
    }
    // what a restore of L cells must reproduce: every DSA layer's rows [0, L) and pooled rows [0, L / 4)
    uint64_t dsa_hash(int64_t L) const {
        uint64_t h = 1469598103934665603ull;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!g.attn_layer(l)) continue;
            h = hash(lat(l), (size_t) L * 512, h);
            h = hash(scl(l), (size_t) L * 16, h);
            h = hash(kg(l), (size_t) L * 512, h);
            h = hash(v.dsa[(size_t) l].pooled, (size_t) (L / 4) * 256, h);
        }
        return h;
    }
};

std::vector<int32_t> tokens(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> t((size_t) n);
    for (auto& x : t) x = (int32_t) (rng() % 1000);
    return t;
}

// a parked image of `s` after `live` with checkpoints at the given lengths (each saved from its own KDA state)
sc::SavedConversation capture(Session& s, const std::vector<int32_t>& live, const std::vector<int64_t>& at,
                              std::vector<uint64_t>& kda_at, std::mt19937& rng) {
    std::vector<sc::ConversationCheckpoint> chain;
    std::string err;
    kda_at.clear();
    for (int64_t L : at) {
        s.scramble_kda(rng);
        sc::ConversationCheckpoint c;
        c.ids.assign(live.begin(), live.begin() + L);
        check(sc::glm_checkpoint_save(c, s.v, err), "checkpoint save");
        c.used = (uint64_t) L;
        kda_at.push_back(s.kda_hash());
        chain.push_back(std::move(c));
    }
    s.scramble(rng);
    const std::vector<sc::ConversationImageKey> no;
    const sc::ConversationView view{live, no, chain, true};
    sc::SavedConversation img;
    check(sc::glm_snapshot_save(img, view, s.v, err), "snapshot save");
    return img;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("glm_conversation_test: no GPU, skipped\n");
        return 77;
    }
    std::mt19937 rng(20261007);
    std::string err;
    const int64_t kCells = 3000, kL = 1777;   // L not a multiple of the pool: a partial last pool
    Session res(kCells, 0), str(kCells, 2400);   // fully resident; streamed with 600 of 750 pages in VRAM
    check(!res.streamed(2) && str.streamed(2), "residency of the two sessions");
    const std::vector<int32_t> live = tokens(kL, 1);
    const std::vector<sc::ConversationImageKey> no;

    // ---- 1. a checkpoint restores the running state bitwise
    {
        res.scramble(rng);
        sc::ConversationCheckpoint c;
        c.ids.assign(live.begin(), live.begin() + 900);
        check(sc::glm_checkpoint_save(c, res.v, err), "checkpoint save");
        sc::GlmConversationSizes z;
        check(sc::glm_conversation_sizes(res.v, z, err), "sizes");
        check(z.n_kda == 4 && z.n_dsa == 2 && c.gdn.size() == 4 * (z.kda_hist + z.kda_state), "running-state bytes");
        check(z.kda_hist + z.kda_state == 9216 + 131072, "KDA bytes per layer (3 x 256 x 3 x 4 + 2 x 128 x 128 x 4)");
        const uint64_t want = res.kda_hash(), dsa = res.dsa_hash(kCells);
        res.scramble_kda(rng);
        check(res.kda_hash() != want, "the scramble changed the state");
        check(sc::glm_checkpoint_restore(c, res.v, err), "checkpoint restore");
        check(res.kda_hash() == want, "checkpoint: running state bitwise");
        check(res.dsa_hash(kCells) == dsa, "checkpoint: DSA cells untouched");
        sc::ConversationCheckpoint bad = c;
        bad.gdn.pop_back();
        check(!sc::glm_checkpoint_validate(bad, res.v, err), "a short running state is refused");
        bad = c;
        bad.tails.assign(4, 0);
        check(!sc::glm_checkpoint_validate(bad, res.v, err), "a Qwen-only field is refused");
    }

    // ---- 2. a parked image restores bitwise, in both residencies and across them
    for (int from = 0; from < 2; ++from)
        for (int to = 0; to < 2; ++to) {
            Session& src = from ? str : res;
            Session& dst = to ? str : res;
            std::vector<uint64_t> kda_at;
            sc::SavedConversation img = capture(src, live, {500, 1200}, kda_at, rng);
            const uint64_t kda = src.kda_hash(), dsa = src.dsa_hash(kL);
            check(img.kv.size() == 2 && img.kv[0].cells == kL && img.kv[0].pooled_rows == kL / 4 &&
                  img.kv[0].format == (sc::kGlmKvFormat | sk::kDsaLatInt8) && img.kv[0].k.size() == (size_t) kL * 512 &&
                  img.kv[0].k_scale.size() == (size_t) kL * 16 && img.kv[0].v.size() == (size_t) kL * 512 &&
                  img.kv[0].pooled.size() == (size_t) (kL / 4) * 256, "image layout");
            check(img.geometry == sc::glm_conversation_key(src.g) && img.geometry[0] == sc::kGlmGeometryTag, "image key");
            dst.scramble(rng);
            check(sc::glm_snapshot_restore(img, dst.v, err) == sc::ConversationRestore::restored, "image restore");
            check(dst.kda_hash() == kda, "image: running state bitwise");
            check(dst.dsa_hash(kL) == dsa, "image: DSA rows and pooled rows bitwise");
            if (to == 1) {   // streamed: every page evicted, so the decode resolves them from the restored host copy
                std::vector<int32_t> table((size_t) dst.v.dsa[2].n_pages);
                CK(cudaMemcpy(table.data(), dst.v.dsa[2].page_table, table.size() * 4, cudaMemcpyDeviceToHost));
                bool none = true;
                for (int32_t t : table) none = none && t == -1;
                check(none, "streamed restore: no page resident");
            }
            // the checkpoints came along: each restores its own running state
            for (size_t k = 0; k < img.checkpoints.size(); ++k) {
                check(sc::glm_checkpoint_restore(img.checkpoints[k], dst.v, err), "image checkpoint restore");
                check(dst.kda_hash() == kda_at[k], "image checkpoint: running state bitwise");
            }
        }

    // ---- 3. the RAM tier
    {
        std::vector<uint64_t> kda_at;
        sc::SavedConversation img = capture(res, live, {500, 1200}, kda_at, rng);
        const uint64_t kda = res.kda_hash(), dsa = res.dsa_hash(kL);
        std::vector<sc::ConversationCheckpoint> chain = img.checkpoints;
        const sc::ConversationView view{live, no, chain, true};
        size_t estimate = 0;
        check(sc::glm_snapshot_bytes(view, res.v, estimate, err), "estimate");
        check(img.bytes() <= estimate && img.bytes() + (1u << 20) > estimate, "the estimate covers the image, closely");
        sc::ConversationCache ram(64ull << 20, 2);
        check(ram.put(std::move(img)), "park in RAM");
        std::vector<int32_t> prompt = live;
        prompt.push_back(7);
        prompt.push_back(8);
        auto m = ram.best(prompt, no, true);
        check(m.tokens == kL && m.live, "RAM: the live tokens match a prompt that extends them");
        std::vector<int32_t> other(live.begin(), live.begin() + 1300);
        other.push_back(9);
        auto m2 = ram.best(other, no, true);
        check(m2.tokens == 1200 && !m2.live, "RAM: a branch matches the deepest checkpoint below it");
        sc::SavedConversation back = ram.take(m.index);
        res.scramble(rng);
        check(sc::glm_snapshot_restore(back, res.v, err) == sc::ConversationRestore::restored, "RAM restore");
        check(res.kda_hash() == kda && res.dsa_hash(kL) == dsa, "RAM: bitwise");
    }

    // ---- 4. the disk tier, unchanged code: write, reopen, match, read, restore
    const fs::path dir = fs::temp_directory_path() / ("glm_conversation_test_" + std::to_string((long long) getpid()));
    for (int keep = 0; keep < 2; ++keep) {
        fs::remove_all(dir);
        std::vector<uint64_t> kda_at;
        sc::SavedConversation img = capture(str, live, {500, 1200}, kda_at, rng);
        const uint64_t kda = str.kda_hash(), dsa = str.dsa_hash(kL);
        const std::string fp = "engine=test arch=glm5-next kv=int8 max_context=3000";
        std::vector<int32_t> prompt = live;
        prompt.push_back(5);
        prompt.push_back(6);
        {
            sc::ConversationDisk d;
            d.set_keep_superseded(keep != 0);
            check(d.open(dir.string(), fp, 0, err), "disk open");
            check(d.store(img, err), "disk store");
        }
        {   // a restart: the directory reopened, the prompt matched by the index, the file read, the image restored
            sc::ConversationDisk d;
            d.set_keep_superseded(keep != 0);
            check(d.open(dir.string(), fp, 0, err) && d.size() == 1, "disk reopen");
            const auto m = d.best(prompt, no, true);
            check(m.tokens == kL && m.live, "disk: the live tokens match a prompt that extends them");
            sc::SavedConversation back;
            check(d.load(m, prompt, no, back, err), "disk load");
            str.scramble(rng);
            check(sc::glm_snapshot_restore(back, str.v, err) == sc::ConversationRestore::restored, "disk restore");
            check(str.kda_hash() == kda && str.dsa_hash(kL) == dsa, "disk: bitwise");
            for (size_t k = 0; k < back.checkpoints.size(); ++k) {
                check(sc::glm_checkpoint_restore(back.checkpoints[k], str.v, err), "disk checkpoint restore");
                check(str.kda_hash() == kda_at[k], "disk checkpoint: running state bitwise");
            }
            // the same conversation a turn later (its live tokens extend this one, which it holds as a checkpoint):
            // it supersedes the first file unless --conversation-cache-disk-keep-superseded
            std::vector<int32_t> longer = live;
            for (int i = 0; i < 40; ++i) longer.push_back((int32_t) (i + 3));
            sc::SavedConversation next = img;
            next.live.ids = longer;
            sc::ConversationCheckpoint ck;
            ck.ids = live;
            ck.gdn = img.live.gdn;
            next.checkpoints.push_back(ck);
            for (auto& kv : next.kv) {   // the first L rows, then the new turn's (zeros here)
                kv.cells = (int64_t) longer.size();
                kv.pooled_rows = kv.cells / 4;
                kv.k.resize((size_t) kv.cells * 512);
                kv.k_scale.resize((size_t) kv.cells * 16);
                kv.v.resize((size_t) kv.cells * 512);
                kv.pooled.resize((size_t) kv.pooled_rows * 256);
            }
            check(sc::glm_snapshot_validate(next, str.v, err), "the next turn's image validates");
            check(d.store(next, err), "disk store of the next turn");
            check(d.size() == (keep ? 2u : 1u), keep ? "keep-superseded keeps both turns" : "the superseded turn is deleted");
        }
        {   // the next turn's file: its checkpoint at L serves the prompt; the rows below L are the first turn's
            sc::ConversationDisk d;
            check(d.open(dir.string(), fp, 0, err) && d.size() == (keep ? 2u : 1u), "disk reopen (2)");
            const auto m = d.best(prompt, no, true);
            check(m.tokens == kL, "disk: a prompt of the first turn matches");
            sc::SavedConversation back;
            check(d.load(m, prompt, no, back, err), "disk load (2)");
            str.scramble(rng);
            check(sc::glm_snapshot_restore(back, str.v, err) == sc::ConversationRestore::restored, "disk restore (2)");
            check(str.dsa_hash(kL) == dsa, "disk (2): the first L rows bitwise");
            const sc::ConversationCheckpoint* c = &back.live;
            for (const auto& k : back.checkpoints) if ((int64_t) k.ids.size() == kL) c = &k;
            check((int64_t) c->ids.size() == kL && sc::glm_checkpoint_restore(*c, str.v, err), "disk (2): turn checkpoint");
            check(str.kda_hash() == kda, "disk (2): the turn checkpoint's running state bitwise");
        }
        // another fingerprint (Qwen's directory, another model) never sees these entries
        sc::ConversationDisk q;
        check(q.open(dir.string(), "engine=test pack=qwen kv=int8 max_context=3000", 0, err) && q.size() == 0,
              "disk: another fingerprint is another directory");
    }
    fs::remove_all(dir);

    // ---- 5. validation
    {
        std::vector<uint64_t> kda_at;
        const sc::SavedConversation img = capture(res, live, {500}, kda_at, rng);
        check(sc::glm_snapshot_validate(img, res.v, err), "a good image validates");
        sc::SavedConversation bad = img;
        bad.geometry = {2560, 48, 4, 128, 16, 48, 4, 10240, 6144, 24, 2, 256, 4, 128, 4, 320, 512, 640};   // Qwen's key
        check(!sc::glm_snapshot_validate(bad, res.v, err), "a Qwen-keyed image is refused");
        check(sc::glm_snapshot_restore(bad, res.v, err) == sc::ConversationRestore::invalid, "... before any write");
        sc::ModelGeometry qwen;   // the defaults are Qwen's
        check(sc::glm_conversation_key(res.g) != bad.geometry && sc::glm_conversation_key(qwen)[0] == sc::kGlmGeometryTag,
              "the GLM key is tagged");
        bad = img;
        bad.kv[1].format = sc::kGlmKvFormat | sk::kDsaLatF16;
        check(!sc::glm_snapshot_validate(bad, res.v, err), "another latent format is refused");
        bad = img;
        bad.kv[0].v.pop_back();
        check(!sc::glm_snapshot_validate(bad, res.v, err), "a short key | gate payload is refused");
        bad = img;
        bad.kv.pop_back();
        check(!sc::glm_snapshot_validate(bad, res.v, err), "a missing DSA layer is refused");
        bad = img;
        bad.checkpoints[0].ids[3] ^= 1;
        check(!sc::glm_snapshot_validate(bad, res.v, err), "a checkpoint off the live tokens is refused");
        bad = img;
        bad.layer_hi = 5;
        check(!sc::glm_snapshot_validate(bad, res.v, err), "another layer range is refused");
        Session small(1000, 0);   // a session too short for the image
        check(!sc::glm_snapshot_validate(img, small.v, err), "an image longer than the context is refused");
    }
    std::printf("glm_conversation_test: %d checks passed\n", checks);
    return 0;
}
