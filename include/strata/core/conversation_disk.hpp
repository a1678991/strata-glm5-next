// On-disk tier for --serve's parked conversations (--conversation-cache-disk PATH).
//
// A parked image (SavedConversation) is also written to a file when it is parked, and the live conversation is
// written at exit, so conversations survive RAM eviction and engine restarts.  On a request, an entry on disk is
// used when it offers a longer prefix than the RAM tier, the live session and the checkpoints; the image is read
// back into RAM and then goes through the same validation and restore as a RAM-parked one.
//
// Writes are asynchronous: begin_store() hands the image to a writer thread (one write in flight) and the request
// goes on; finish() takes the image back once the file is complete, so the caller can park it in the RAM tier.
// While it is in flight the image is only READ by the writer; the caller must not touch it until finish().
//
// Matching uses an in-memory index instead of the tokens: every head (the live tokens and each checkpoint) is
// kept as (length, 64-bit prefix hash, image keys), and a prompt is hashed once, so a lookup costs one pass over
// the prompt plus one hash-table probe per distinct head length.  Files carry the same compact header, so opening
// the directory reads a few hundred bytes per entry.  A hit is confirmed against the full tokens after loading.
//
// Files live in a subdirectory named after a fingerprint of everything that changes the layout or meaning of a
// saved state (model files, KV format, context, ...), so an image is never offered to another configuration;
// conversation_snapshot_validate still checks every layer before anything is written to the GPU.  Each file
// carries a checksum over its payload.  One GPU only (an image with stage parts is not written).
//
// Eviction follows the RAM tier (ConversationCache): entries are ordered least recently active first - an entry
// becomes most recent when it is written (parked) or read back (restored), as a RAM entry does when it is put
// back - and the oldest go first while the directory is over its size limit.  By default a newly parked image
// also deletes the copies it supersedes (the same conversation a turn back, the RAM tier's #342 rule);
// set_keep_superseded(true) keeps them, so only the size limit removes entries.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::core {

/// The hash of a token prefix, extended one token at a time (h(0) = seed): the index's key with the length.
inline uint64_t conversation_prefix_hash_step(uint64_t h, int64_t token) {
    h = (h ^ (uint64_t) (uint32_t) (int32_t) token) * 0x9e3779b97f4a7c15ull;
    return h ^ (h >> 29);
}
constexpr uint64_t kConversationPrefixHashSeed = 0x243f6a8885a308d3ull;

class ConversationDisk {
public:
    struct Match {
        std::string path;
        int64_t tokens = 0;
        bool live = false;
        size_t size = 0;
    };
    enum class Store { started, unchanged, refused };

    ConversationDisk() = default;
    ConversationDisk(const ConversationDisk&) = delete;
    ConversationDisk& operator=(const ConversationDisk&) = delete;
    ~ConversationDisk();

    // Scans `dir/<fingerprint hash>/`, indexes the valid entries (headers only) and evicts past `limit` bytes
    // (0 = no limit).  Returns false (tier disabled) when the directory cannot be created.
    bool open(const std::string& dir, const std::string& fingerprint, size_t limit, std::string& error);
    bool enabled() const { return !dir_.empty(); }
    // Keep the copies a newly parked image supersedes instead of deleting them (default: delete, as RAM does).
    void set_keep_superseded(bool keep) { keep_superseded_ = keep; }
    bool keep_superseded() const { return keep_superseded_; }
    const std::string& dir() const { return dir_; }
    size_t size() const { return entries_.size(); }
    size_t bytes() const;

    // The entry with the longest usable prefix of `prompt` (the RAM tier's rule: the whole head is a prefix, the
    // images below it match, and it is shorter than the prompt).  Ties prefer the most recently used entry.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const;
    // Reads and checks the entry of `m`, then confirms the hit against the full tokens it holds (a prefix-hash
    // collision is a miss, not a wrong restore).  A damaged or foreign file is removed.
    template<class Token>
    bool load(const Match& m, const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
              SavedConversation& image, std::string& error);

    // Asynchronous write.  `started`: the writer owns the image until finish() returns it.  `unchanged`: an entry
    // with the same heads exists (it is marked used); `refused`: not writable (error says why).  In both of those
    // the image stays with the caller.  Only one write is in flight: the caller finishes the previous one first.
    Store begin_store(SavedConversation&& image, std::string& error);
    bool busy() const { return job_ != nullptr; }
    size_t in_flight_bytes() const;
    // The prefix the image in flight offers this prompt (0: none, or nothing in flight).
    template<class Token>
    int64_t in_flight_match(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                            bool cvec) const;
    // The image of the write in flight once it is complete (`wait`: block until it is), indexing the file when it
    // was written.  nullopt: nothing in flight, or not complete yet and `wait` is false.
    std::optional<SavedConversation> finish(bool wait);

    // Synchronous write (the live conversation at exit; tests).
    bool store(const SavedConversation& image, std::string& error);

private:
    struct Head {
        int64_t len = 0;
        uint64_t hash = 0;
        std::vector<ConversationImageKey> imgs;
    };
    struct Entry {
        std::string path;
        size_t size = 0;
        bool cvec = true;
        uint64_t used = 0;           // LRU stamp
        std::vector<Head> heads;     // [0] the live tokens, then the checkpoints
    };
    using EntryIt = std::list<Entry>::iterator;
    struct Job {
        SavedConversation image;
        Entry entry;
        std::string tmp, error;
        bool ok = false;
        double ms = 0;
        std::atomic<bool> done{false};
        std::thread thread;
    };

    static std::vector<Head> heads_of(const SavedConversation& image);
    static bool images_match(const Head& h, const std::vector<ConversationImageKey>& images);
    Store prepare(const SavedConversation& image, Entry& entry, std::string& error);
    bool write_file(const SavedConversation& image, const Entry& entry, const std::string& tmp, std::string& error) const;
    bool commit(Entry&& entry, const std::string& tmp, std::string& error);
    void add(Entry&& entry);
    void touch(EntryIt it);
    void drop_superseded(const std::vector<Head>& heads, bool cvec);
    void evict(size_t incoming);
    void remove(EntryIt it);
    bool load_file(const std::string& path, uint64_t size, SavedConversation& image, std::string& error) const;

    std::string dir_, fingerprint_;
    size_t limit_ = 0;
    bool keep_superseded_ = false;
    uint64_t serial_ = 0, clock_ = 0;
    std::list<Entry> entries_;   // least recently used first
    // the index: (length, prefix hash) -> the heads with it; and how many heads have each length
    std::unordered_map<uint64_t, std::vector<std::pair<EntryIt, size_t>>> index_;
    std::map<int64_t, size_t> lengths_;
    std::unique_ptr<Job> job_;
};

inline uint64_t conversation_index_key(int64_t len, uint64_t hash) {
    return (hash ^ ((uint64_t) len * 0xff51afd7ed558ccdull)) * 0xc4ceb9fe1a85ec53ull;
}

inline bool ConversationDisk::images_match(const Head& h, const std::vector<ConversationImageKey>& images) {
    // conversation_prefix's image rule: the request's images below the head are exactly the head's, in order
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= h.len) continue;
        if (j == h.imgs.size() || !(h.imgs[j++] == image)) return false;
    }
    return j == h.imgs.size();
}

template<class Token>
ConversationDisk::Match ConversationDisk::best(const std::vector<Token>& prompt,
                                               const std::vector<ConversationImageKey>& images, bool cvec) const {
    Match m;
    const int64_t n = (int64_t) prompt.size();
    if (lengths_.empty() || n < 2) return m;
    // prefix hashes up to the longest usable length (the last prompt token always starts the next window)
    auto top = lengths_.lower_bound(n);
    if (top == lengths_.begin()) return m;
    const int64_t max_len = std::prev(top)->first;
    std::vector<uint64_t> ph((size_t) max_len + 1);
    ph[0] = kConversationPrefixHashSeed;
    for (int64_t i = 0; i < max_len; ++i) ph[(size_t) i + 1] = conversation_prefix_hash_step(ph[(size_t) i], prompt[(size_t) i]);
    for (auto it = std::make_reverse_iterator(top); it != lengths_.rend(); ++it) {
        const int64_t len = it->first;
        auto hit = index_.find(conversation_index_key(len, ph[(size_t) len]));
        if (hit == index_.end()) continue;
        const Entry* best_e = nullptr;
        size_t best_h = 0;
        for (const auto& [e, h] : hit->second) {
            const Head& head = e->heads[h];
            if (head.len != len || head.hash != ph[(size_t) len] || e->cvec != cvec || !images_match(head, images)) continue;
            if (!best_e || e->used > best_e->used) { best_e = &*e; best_h = h; }
        }
        if (best_e) return {best_e->path, len, best_h == 0, best_e->size};
    }
    return m;
}

template<class Token>
bool ConversationDisk::load(const Match& m, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images, SavedConversation& image,
                            std::string& error) {
    auto it = entries_.begin();
    while (it != entries_.end() && it->path != m.path) ++it;
    if (it == entries_.end()) { error = "no such entry"; return false; }
    SavedConversation out;
    if (!load_file(it->path, it->size, out, error)) {
        std::fprintf(stderr, "strata serve: conversation cache disk: removing %s (%s)\n", it->path.c_str(), error.c_str());
        remove(it);
        return false;
    }
    int64_t got = conversation_prefix(out.live, prompt, images);
    for (const auto& c : out.checkpoints) got = std::max(got, conversation_prefix(c, prompt, images));
    if (got != m.tokens) {
        error = "the full tokens do not confirm the index hit (" + std::to_string(got) + " of " +
                std::to_string(m.tokens) + ")";
        return false;
    }
    touch(it);
    image = std::move(out);
    return true;
}

template<class Token>
int64_t ConversationDisk::in_flight_match(const std::vector<Token>& prompt,
                                          const std::vector<ConversationImageKey>& images, bool cvec) const {
    if (!job_ || job_->image.cvec != cvec) return 0;
    int64_t n = conversation_prefix(job_->image.live, prompt, images);
    for (const auto& c : job_->image.checkpoints) n = std::max(n, conversation_prefix(c, prompt, images));
    return n;
}

}  // namespace strata::core
