// CPU-only checks of the conversation cache's on-disk tier: round trip, matching, damage, isolation, eviction.
#include "strata/core/conversation_disk.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <string>
#include <unistd.h>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

std::vector<int32_t> seq(int32_t from, int32_t n) {
    std::vector<int32_t> v;
    for (int32_t i = 0; i < n; ++i) v.push_back(from + i);
    return v;
}

ConversationBuffer filled(size_t n, uint8_t salt) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t m, size_t at) {
        for (size_t i = 0; i < m; ++i) p[i] = (uint8_t) ((at + i) * 131 + salt);
        return true;
    });
    return b;
}

SavedConversation image(std::vector<int32_t> ids, std::vector<size_t> check_lens, size_t kv_bytes, uint8_t salt) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = (int64_t) (i * 7 + 1);
    s.layer_lo = 0; s.layer_hi = 48;
    s.live.ids = ids;
    s.live.gdn.assign(1000, salt);
    s.live.ple.assign(17, (uint8_t) (salt + 1));
    s.live.used = 5;
    for (size_t n : check_lens) {
        ConversationCheckpoint c;
        c.ids.assign(ids.begin(), ids.begin() + (std::ptrdiff_t) n);
        c.imgs.push_back({1, 0xabcdef});
        c.gdn.assign(300, (uint8_t) n);
        c.tails.assign(9, 3);
        c.used = n;
        s.checkpoints.push_back(std::move(c));
    }
    ConversationKv kv;
    kv.format = 2; kv.cells = 256; kv.heads = 2; kv.head_dim = 128; kv.page_size = 64; kv.pooled_rows = 3; kv.idx_dim = 8;
    kv.k = filled(kv_bytes, salt);
    kv.v = filled(kv_bytes + 3, (uint8_t) (salt + 9));   // not a multiple of 8: the checksum's tail
    kv.pooled = filled(77, salt);
    s.kv.push_back(std::move(kv));
    s.cvec = true;
    return s;
}

bool equal(const SavedConversation& a, const SavedConversation& b) {
    auto ck = [](const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
        return x.ids == y.ids && x.imgs == y.imgs && x.gdn == y.gdn && x.ple == y.ple && x.tails == y.tails &&
               x.dead == y.dead && x.block_pos == y.block_pos && x.used == y.used;
    };
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !ck(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i) if (!ck(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled))
            return false;
    }
    return true;
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / ("strata-conv-disk-test-" + std::to_string(::getpid()));
    fs::remove_all(root);
    std::string err;
    const std::vector<ConversationImageKey> no_imgs;

    // round trip, across a "restart" (a second instance indexing the same directory)
    const auto a = image(seq(100, 500), {50, 200}, 40u << 20, 1);   // 40 MiB: several buffer segments
    {
        ConversationDisk d;
        check(d.open(root.string(), "model-A ctx=131072", 0, err), "open");
        check(d.store(a, err), "store");
        check(d.size() == 1, "one entry");
    }
    ConversationDisk d;
    check(d.open(root.string(), "model-A ctx=131072", 0, err), "reopen");
    check(d.size() == 1, "entry indexed after restart");
    auto prompt = seq(100, 600);
    auto m = d.best(prompt, no_imgs, true);
    check(m.tokens == 500 && m.live, "live prefix matches");
    check(d.best(prompt, no_imgs, false).tokens == 0, "control-vector state must match");
    check(d.best(seq(100, 500), no_imgs, true).tokens == 0, "the last prompt token is never covered");
    check(d.best(seq(100, 120), no_imgs, true).tokens == 0, "a checkpoint with images needs them");
    SavedConversation b;
    check(d.load(m, prompt, no_imgs, b, err), "load");
    check(equal(a, b), "byte-identical round trip");
    // a checkpoint matches through the index with its images; the live head (no images) does not
    const std::vector<ConversationImageKey> with_img{{1, 0xabcdef}};
    auto mc = d.best(prompt, with_img, true);
    check(mc.tokens == 200 && !mc.live, "deepest checkpoint with matching images");
    SavedConversation bc;
    check(d.load(mc, prompt, with_img, bc, err), "load confirms the checkpoint hit");
    check(d.best(prompt, std::vector<ConversationImageKey>{{1, 0xabcdee}}, true).tokens == 0, "other image: no match");
    // a prompt that diverges inside the conversation matches only the heads before the divergence
    auto diverged = prompt;
    diverged[300] = -7;
    check(d.best(diverged, with_img, true).tokens == 200, "divergence after a checkpoint");
    diverged[40] = -7;
    check(d.best(diverged, with_img, true).tokens == 0, "divergence before every head");

    // another configuration never sees it
    {
        ConversationDisk other;
        check(other.open(root.string(), "model-B ctx=131072", 0, err), "open other");
        check(other.size() == 0, "fingerprint isolates configurations");
    }

    // the same conversation a turn later supersedes it
    auto a2 = image(seq(100, 800), {50, 200, 500}, 1u << 20, 2);
    check(d.store(a2, err), "store next turn");
    check(d.size() == 1, "older turn dropped");
    check(d.best(seq(100, 900), no_imgs, true).tokens == 800, "newer turn matches");

    // a different conversation stays; LRU eviction by size
    const auto c = image(seq(5000, 300), {}, 1u << 20, 3);
    check(d.store(c, err), "store other conversation");
    check(d.size() == 2, "two conversations");
    {
        ConversationDisk small;
        check(small.open(root.string(), "model-A ctx=131072", d.bytes() - 1, err), "open with a smaller limit");
        check(small.size() == 1, "evicted down to the limit");
        check(small.best(seq(5000, 400), no_imgs, true).tokens == 300, "the most recent entry survives");
    }

    // damage: flip a payload byte; load must fail and remove the file
    {
        ConversationDisk e;
        check(e.open(root.string(), "model-A ctx=131072", 0, err), "open for damage");
        const auto hit = e.best(seq(5000, 400), no_imgs, true);
        check(hit.tokens == 300, "entry present");
        {
            std::fstream f(hit.path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp((std::streamoff) (hit.size / 2));
            char ch = 0;
            f.read(&ch, 1);
            f.seekp((std::streamoff) (hit.size / 2));
            ch ^= 0x40;
            f.write(&ch, 1);
        }
        SavedConversation x;
        check(!e.load(hit, seq(5000, 400), no_imgs, x, err), "damaged entry rejected");
        check(!fs::exists(hit.path), "damaged entry removed");
        check(e.size() == 0, "index updated");
    }

    // asynchronous write: the writer owns the image, the index has it once finished
    {
        ConversationDisk e;
        check(e.open((root / "async").string(), "model-A ctx=131072", 0, err), "open async");
        auto img = image(seq(9000, 700), {100, 400}, 24u << 20, 6);
        const auto copy = img;
        check(e.begin_store(std::move(img), err) == ConversationDisk::Store::started, "async write started");
        check(e.busy(), "in flight");
        check(e.in_flight_match(seq(9000, 800), no_imgs, true) == 700, "in-flight image matches");
        check(e.begin_store(image(seq(1, 10), {}, 64, 7), err) == ConversationDisk::Store::refused, "one write in flight");
        auto back = e.finish(true);
        check(back && equal(*back, copy), "finish returns the image unchanged");
        check(!e.busy() && e.size() == 1, "indexed after finish");
        check(!e.finish(true), "nothing in flight");
        check(e.best(seq(9000, 800), no_imgs, true).tokens == 700, "async entry matches");
        auto again = copy;
        check(e.begin_store(std::move(again), err) == ConversationDisk::Store::unchanged, "unchanged entry kept");
        check(!e.busy(), "no write for an unchanged entry");
        ConversationDisk r;
        check(r.open((root / "async").string(), "model-A ctx=131072", 0, err) && r.size() == 1, "async entry after restart");
    }

    // keep-superseded: a later turn does not delete the earlier one; the size limit removes the least recently
    // active entry (parked or restored), as the RAM tier does
    {
        ConversationDisk e;
        e.set_keep_superseded(true);
        check(e.open((root / "keep").string(), "model-A ctx=131072", 0, err), "open keep");
        const auto t1 = image(seq(100, 500), {50, 200}, 1u << 20, 11);
        const auto t2 = image(seq(100, 800), {50, 200, 500}, 1u << 20, 12);
        check(e.store(t1, err) && e.store(t2, err), "store two turns");
        check(e.size() == 2, "the earlier turn is kept");
        const std::vector<ConversationImageKey> img{{1, 0xabcdef}};
        auto old_hit = e.best(seq(100, 501), no_imgs, true);
        check(old_hit.tokens == 500 && old_hit.live, "the earlier turn's live tokens still match");
        auto shared = e.best(seq(100, 450), img, true);
        check(shared.tokens == 200 && shared.path != old_hit.path, "a shared checkpoint resolves to the most recent copy");
        // restoring t1 makes it the most recent; a third entry over the limit then evicts t2 first
        SavedConversation x;
        check(e.load(old_hit, seq(100, 501), no_imgs, x, err), "restore the earlier turn");
        ConversationDisk f;
        f.set_keep_superseded(true);
        check(f.open((root / "keep").string(), "model-A ctx=131072", e.bytes() + 4096, err), "reopen: a third entry needs an eviction");
        check(f.store(image(seq(7000, 300), {}, 1u << 20, 13), err), "store a third conversation");
        check(f.size() == 2, "one evicted");
        check(f.best(seq(100, 501), no_imgs, true).tokens == 500, "the restored (recently active) turn stays");
        check(f.best(seq(100, 900), no_imgs, true).tokens < 800, "the least recently active turn went");
    }

    // index scale: many conversations, one lookup
    {
        ConversationDisk e;
        check(e.open((root / "many").string(), "model-A ctx=131072", 0, err), "open many");
        for (int i = 0; i < 200; ++i) check(e.store(image(seq(100000 * (i + 1), 4000), {1000, 2000, 3000}, 64, 8), err), "store many");
        auto q = seq(100000 * 137, 5000);
        const auto t0 = std::chrono::steady_clock::now();
        auto hit = e.best(q, no_imgs, true);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        check(hit.tokens == 4000 && hit.live, "the right conversation among 200");
        std::printf("index lookup over %zu entries: %.3f ms\n", e.size(), ms);
    }

    // layer-split images are refused
    {
        auto s = image(seq(1, 10), {}, 64, 4);
        s.stage_images.push_back(image(seq(1, 10), {}, 64, 5));
        check(!d.store(s, err), "layer-split image refused");
    }

    fs::remove_all(root);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
