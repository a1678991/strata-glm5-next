#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <system_error>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;

constexpr uint32_t kMagic = 0x31444353;   // "SCD1"
constexpr uint32_t kVersion = 2;          // v2: compact header (length, prefix hash, images per head)
constexpr size_t kIoBuffer = 8u << 20;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

// A 64-bit running checksum over a byte stream, independent of how the stream is split into calls: a damaged
// file whose length is intact (renamed into place before its data reached the disk, then a power loss) is caught.
class Checksum {
public:
    void update(const void* data, size_t n) {
        const auto* p = static_cast<const uint8_t*>(data);
        while (n > 0 && carry_n_ > 0) {
            carry_[carry_n_++] = *p++; --n;
            if (carry_n_ == 8) { mix(carry_); carry_n_ = 0; }
        }
        for (; n >= 8; p += 8, n -= 8) mix(p);
        while (n > 0) { carry_[carry_n_++] = *p++; --n; }
    }
    uint64_t value() const {
        uint64_t h = h_;
        for (size_t i = 0; i < carry_n_; ++i) h = (h ^ carry_[i]) * 0x100000001b3ull;
        return h ^ (uint64_t) carry_n_;
    }
private:
    void mix(const uint8_t* p) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        h_ = (h_ ^ w) * 0x9e3779b97f4a7c15ull;
        h_ ^= h_ >> 29;
    }
    uint64_t h_ = 0x243f6a8885a308d3ull;
    uint8_t carry_[8] = {};
    size_t carry_n_ = 0;
};

struct FileCloser { void operator()(FILE* f) const { if (f) std::fclose(f); } };
using File = std::unique_ptr<FILE, FileCloser>;

// Cache files must not push the model's pages out of the page cache: written files are flushed and their pages
// dropped, read files' pages are dropped after the read.
void drop_pages(FILE* f) {
#if defined(__linux__)
    std::fflush(f);
    posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
#else
    (void) f;
#endif
}

struct Writer {
    explicit Writer(FILE* file) : f(file) {}
    FILE* f;
    bool ok = true;
    bool summed = false;
    Checksum sum;
    void raw(const void* p, size_t n) {
        if (!ok || n == 0) return;
        if (std::fwrite(p, 1, n, f) != n) { ok = false; return; }
        if (summed) sum.update(p, n);
    }
    template<class T> void pod(const T& v) { raw(&v, sizeof(v)); }
    void bytes(const std::vector<uint8_t>& v) { pod<uint64_t>(v.size()); raw(v.data(), v.size()); }
    void ids(const std::vector<int32_t>& v) { pod<uint64_t>(v.size()); raw(v.data(), v.size() * sizeof(int32_t)); }
    void imgs(const std::vector<ConversationImageKey>& v) {
        pod<uint64_t>(v.size());
        for (const auto& k : v) { pod(k.start); pod(k.hash); }
    }
    void buffer(const ConversationBuffer& b) {
        pod<uint64_t>(b.size());
        if (!b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t) { raw(p, n); return ok; })) ok = false;
    }
};

struct Reader {
    Reader(FILE* file, uint64_t bound) : f(file), limit(bound) {}
    FILE* f;
    uint64_t limit;   // no single field can exceed the file itself
    bool ok = true;
    bool summed = false;
    Checksum sum;
    void raw(void* p, size_t n) {
        if (!ok || n == 0) return;
        if (std::fread(p, 1, n, f) != n) { ok = false; return; }
        if (summed) sum.update(p, n);
    }
    template<class T> void pod(T& v) { raw(&v, sizeof(v)); }
    uint64_t count(uint64_t unit) {
        uint64_t n = 0;
        pod(n);
        if (!ok || unit == 0 || n > limit / unit) { ok = false; return 0; }
        return n;
    }
    void bytes(std::vector<uint8_t>& v) {
        const uint64_t n = count(1);
        if (!ok) return;
        v.resize((size_t) n);
        raw(v.data(), (size_t) n);
    }
    void ids(std::vector<int32_t>& v) {
        const uint64_t n = count(sizeof(int32_t));
        if (!ok) return;
        v.resize((size_t) n);
        raw(v.data(), (size_t) n * sizeof(int32_t));
    }
    void imgs(std::vector<ConversationImageKey>& v) {
        const uint64_t n = count(16);
        if (!ok) return;
        v.resize((size_t) n);
        for (auto& k : v) { pod(k.start); pod(k.hash); }
    }
    void buffer(ConversationBuffer& b) {
        const uint64_t n = count(1);
        if (!ok) return;
        b.resize((size_t) n);
        if (!b.visit(0, (size_t) n, [&](uint8_t* p, size_t m, size_t) { raw(p, m); return ok; })) ok = false;
    }
};

void write_checkpoint(Writer& w, const ConversationCheckpoint& c) {
    w.ids(c.ids); w.imgs(c.imgs);
    w.bytes(c.gdn); w.bytes(c.ple); w.bytes(c.tails); w.bytes(c.dead); w.bytes(c.block_pos);
    w.pod(c.used);
}

void read_checkpoint(Reader& r, ConversationCheckpoint& c) {
    r.ids(c.ids); r.imgs(c.imgs);
    r.bytes(c.gdn); r.bytes(c.ple); r.bytes(c.tails); r.bytes(c.dead); r.bytes(c.block_pos);
    r.pod(c.used);
}

void sync_dir(const std::string& dir) {
#if defined(__linux__)
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
#else
    (void) dir;
#endif
}

uint64_t prefix_hash(const std::vector<int32_t>& ids) {
    uint64_t h = kConversationPrefixHashSeed;
    for (int32_t t : ids) h = conversation_prefix_hash_step(h, t);
    return h;
}

}  // namespace

ConversationDisk::~ConversationDisk() {
    if (job_ && job_->thread.joinable()) job_->thread.join();
}

std::vector<ConversationDisk::Head> ConversationDisk::heads_of(const SavedConversation& image) {
    std::vector<Head> heads;
    heads.push_back({(int64_t) image.live.ids.size(), prefix_hash(image.live.ids), image.live.imgs});
    for (const auto& c : image.checkpoints) heads.push_back({(int64_t) c.ids.size(), prefix_hash(c.ids), c.imgs});
    return heads;
}

// magic, version, fingerprint, cvec, then the heads: length, prefix hash, images
static bool read_header(Reader& r, const std::string& fingerprint, bool& cvec, std::vector<int64_t>& lens,
                        std::vector<uint64_t>& hashes, std::vector<std::vector<ConversationImageKey>>& imgs) {
    uint32_t magic = 0, version = 0;
    r.pod(magic); r.pod(version);
    if (!r.ok || magic != kMagic || version != kVersion) return false;
    const uint64_t n_fp = r.count(1);
    if (!r.ok || n_fp != fingerprint.size()) return false;
    std::string fp((size_t) n_fp, '\0');
    r.raw(fp.data(), fp.size());
    if (!r.ok || fp != fingerprint) return false;
    uint8_t c = 0;
    r.pod(c);
    cvec = c != 0;
    const uint64_t n = r.count(32);
    if (!r.ok || n == 0) return false;
    lens.resize((size_t) n); hashes.resize((size_t) n); imgs.resize((size_t) n);
    for (size_t i = 0; i < n; ++i) { r.pod(lens[i]); r.pod(hashes[i]); r.imgs(imgs[i]); }
    return r.ok && lens[0] > 0;
}

bool ConversationDisk::open(const std::string& dir, const std::string& fingerprint, size_t limit, std::string& error) {
    char sub[32];
    std::snprintf(sub, sizeof(sub), "%016llx", (unsigned long long) fnv1a(fingerprint.data(), fingerprint.size()));
    std::error_code ec;
    const fs::path d = fs::path(dir) / sub;
    fs::create_directories(d, ec);
    if (ec) {
        error = "cannot create " + d.string() + ": " + ec.message();
        return false;
    }
    dir_ = d.string();
    fingerprint_ = fingerprint;
    limit_ = limit;
    if (File f{std::fopen((d / "fingerprint.txt").c_str(), "w")}) std::fputs(fingerprint.c_str(), f.get());

    struct Found { Entry e; fs::file_time_type t; };
    std::vector<Found> found;
    for (const auto& de : fs::directory_iterator(d, ec)) {
        const fs::path p = de.path();
        if (p.extension() == ".tmp") { fs::remove(p, ec); continue; }   // an interrupted write
        if (p.extension() != ".scd") continue;
        const auto size = fs::file_size(p, ec);
        if (ec) continue;
        File f{std::fopen(p.c_str(), "rb")};
        if (!f) continue;
        Reader r{f.get(), (uint64_t) size};
        Found x;
        std::vector<int64_t> lens;
        std::vector<uint64_t> hashes;
        std::vector<std::vector<ConversationImageKey>> imgs;
        if (!read_header(r, fingerprint_, x.e.cvec, lens, hashes, imgs)) {
            std::fprintf(stderr, "strata serve: conversation cache disk: ignoring invalid entry %s\n", p.c_str());
            continue;
        }
        drop_pages(f.get());
        for (size_t i = 0; i < lens.size(); ++i) x.e.heads.push_back({lens[i], hashes[i], std::move(imgs[i])});
        x.e.path = p.string();
        x.e.size = (size_t) size;
        x.t = fs::last_write_time(p, ec);
        found.push_back(std::move(x));
    }
    std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
    for (auto& x : found) add(std::move(x.e));
    evict(0);
    return true;
}

size_t ConversationDisk::bytes() const {
    size_t n = 0;
    for (const auto& e : entries_) n += e.size;
    return n;
}

size_t ConversationDisk::in_flight_bytes() const { return job_ ? job_->image.bytes() : 0; }

void ConversationDisk::add(Entry&& entry) {
    entry.used = ++clock_;
    entries_.push_back(std::move(entry));
    const EntryIt it = std::prev(entries_.end());
    for (size_t h = 0; h < it->heads.size(); ++h) {
        index_[conversation_index_key(it->heads[h].len, it->heads[h].hash)].push_back({it, h});
        ++lengths_[it->heads[h].len];
    }
}

void ConversationDisk::touch(EntryIt it) {
    it->used = ++clock_;
    entries_.splice(entries_.end(), entries_, it);   // iterators stay valid
    std::error_code ec;
    fs::last_write_time(it->path, fs::file_time_type::clock::now(), ec);   // least recently used across restarts too
}

void ConversationDisk::remove(EntryIt it) {
    for (size_t h = 0; h < it->heads.size(); ++h) {
        const uint64_t key = conversation_index_key(it->heads[h].len, it->heads[h].hash);
        auto& refs = index_[key];
        refs.erase(std::remove_if(refs.begin(), refs.end(), [&](const auto& r) { return r.first == it; }), refs.end());
        if (refs.empty()) index_.erase(key);
        if (--lengths_[it->heads[h].len] == 0) lengths_.erase(it->heads[h].len);
    }
    std::error_code ec;
    fs::remove(it->path, ec);
    entries_.erase(it);
}

void ConversationDisk::evict(size_t incoming) {
    while (limit_ != 0 && !entries_.empty() && bytes() + incoming > limit_) {
        std::fprintf(stderr, "strata serve: conversation cache disk: evict %s (%zu MiB)\n",
                     entries_.front().path.c_str(), entries_.front().size >> 20);
        remove(entries_.begin());
    }
}

void ConversationDisk::drop_superseded(const std::vector<Head>& heads, bool cvec) {
    // The RAM tier's #342 rule: an entry whose deepest checkpoint the outgoing chain holds is the same
    // conversation a turn back.  An entry with exactly the outgoing live tokens is the same conversation too.
    auto same = [](const Head& a, const Head& b) { return a.len == b.len && a.hash == b.hash && a.imgs == b.imgs; };
    for (auto it = entries_.begin(); it != entries_.end();) {
        const Entry& e = *it;
        const Head* deepest = nullptr;
        for (size_t i = 1; i < e.heads.size(); ++i)
            if (!deepest || e.heads[i].len > deepest->len) deepest = &e.heads[i];
        bool drop = e.cvec == cvec && same(e.heads[0], heads[0]);
        if (!drop && e.cvec == cvec && deepest && deepest->len > 0)
            for (const auto& h : heads) drop = drop || same(*deepest, h);
        if (drop) {
            std::fprintf(stderr, "strata serve: conversation cache disk: drop superseded %s\n", e.path.c_str());
            auto next = std::next(it);
            remove(it);
            it = next;
        } else {
            ++it;
        }
    }
}

// The index work of a write, on the caller's thread: refuse, recognize an unchanged entry, or make room and
// name the file.
ConversationDisk::Store ConversationDisk::prepare(const SavedConversation& image, Entry& entry, std::string& error) {
    if (!enabled()) { error = "disabled"; return Store::refused; }
    if (!image.stage_images.empty() || !image.live.stage_parts.empty()) {
        error = "layer-split images are not written to disk";
        return Store::refused;
    }
    for (const auto& c : image.checkpoints)
        if (!c.stage_parts.empty()) { error = "layer-split checkpoints are not written to disk"; return Store::refused; }
    if (image.live.ids.empty()) { error = "empty conversation"; return Store::refused; }

    entry.heads = heads_of(image);
    entry.cvec = image.cvec;
    // the same conversation, unchanged (restored and parked again): keep the file, mark it used
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->cvec != entry.cvec || it->heads.size() != entry.heads.size()) continue;
        bool same = true;
        for (size_t i = 0; same && i < entry.heads.size(); ++i)
            same = it->heads[i].len == entry.heads[i].len && it->heads[i].hash == entry.heads[i].hash &&
                   it->heads[i].imgs == entry.heads[i].imgs;
        if (!same) continue;
        touch(it);
        return Store::unchanged;
    }
    if (!keep_superseded_) drop_superseded(entry.heads, entry.cvec);
    evict(image.bytes());
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    char name[64];
    std::snprintf(name, sizeof(name), "%020lld-%llu.scd",
                  (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(now).count(),
                  (unsigned long long) ++serial_);
    entry.path = (fs::path(dir_) / name).string();
    return Store::started;
}

// The file itself: thread-safe (reads only the image and the immutable fingerprint).
bool ConversationDisk::write_file(const SavedConversation& image, const Entry& entry, const std::string& tmp,
                                  std::string& error) const {
    std::vector<char> buf(kIoBuffer);   // outlives the FILE that uses it
    File f{std::fopen(tmp.c_str(), "wb")};
    if (!f) { error = "cannot create " + tmp; return false; }
    std::setvbuf(f.get(), buf.data(), _IOFBF, buf.size());
    Writer w{f.get()};
    w.pod(kMagic); w.pod(kVersion);
    w.pod<uint64_t>(fingerprint_.size()); w.raw(fingerprint_.data(), fingerprint_.size());
    w.pod<uint8_t>(image.cvec ? 1 : 0);
    w.pod<uint64_t>(entry.heads.size());
    for (const auto& h : entry.heads) { w.pod(h.len); w.pod(h.hash); w.imgs(h.imgs); }
    w.summed = true;   // the payload
    for (int64_t x : image.geometry) w.pod(x);
    w.pod(image.layer_lo); w.pod(image.layer_hi);
    write_checkpoint(w, image.live);
    w.pod<uint64_t>(image.checkpoints.size());
    for (const auto& c : image.checkpoints) write_checkpoint(w, c);
    w.pod<uint64_t>(image.kv.size());
    for (const auto& k : image.kv) {
        w.pod(k.format); w.pod(k.cells); w.pod(k.heads); w.pod(k.head_dim); w.pod(k.page_size);
        w.pod(k.pooled_rows); w.pod(k.idx_dim);
        w.buffer(k.k); w.buffer(k.v); w.buffer(k.k_scale); w.buffer(k.v_scale); w.buffer(k.pooled);
    }
    w.summed = false;
    w.pod(w.sum.value());
    bool ok = w.ok && std::fflush(f.get()) == 0;
#if defined(__linux__)
    ok = ok && ::fdatasync(fileno(f.get())) == 0;
#endif
    if (ok) drop_pages(f.get());
    if (std::fclose(f.release()) != 0) ok = false;
    if (!ok) {
        std::error_code ec;
        fs::remove(tmp, ec);
        error = "write failed: " + tmp;
    }
    return ok;
}

// Renames the finished file into place and indexes it (the caller's thread).
bool ConversationDisk::commit(Entry&& entry, const std::string& tmp, std::string& error) {
    std::error_code ec;
    fs::rename(tmp, entry.path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        error = "rename failed: " + entry.path;
        return false;
    }
    sync_dir(dir_);
    entry.size = (size_t) fs::file_size(entry.path, ec);
    add(std::move(entry));
    evict(0);
    return true;
}

bool ConversationDisk::store(const SavedConversation& image, std::string& error) {
    Entry entry;
    switch (prepare(image, entry, error)) {
        case Store::refused: return false;
        case Store::unchanged: return true;
        case Store::started: break;
    }
    const std::string tmp = entry.path + ".tmp";
    return write_file(image, entry, tmp, error) && commit(std::move(entry), tmp, error);
}

ConversationDisk::Store ConversationDisk::begin_store(SavedConversation&& image, std::string& error) {
    if (job_) { error = "a write is still in flight"; return Store::refused; }
    Entry entry;
    const Store s = prepare(image, entry, error);
    if (s != Store::started) return s;
    job_ = std::make_unique<Job>();
    job_->image = std::move(image);
    job_->entry = std::move(entry);
    job_->tmp = job_->entry.path + ".tmp";
    Job* job = job_.get();
    job->thread = std::thread([this, job] {
        const auto t0 = std::chrono::steady_clock::now();
        job->ok = write_file(job->image, job->entry, job->tmp, job->error);
        job->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        job->done.store(true, std::memory_order_release);
    });
    return Store::started;
}

std::optional<SavedConversation> ConversationDisk::finish(bool wait) {
    if (!job_) return std::nullopt;
    if (!wait && !job_->done.load(std::memory_order_acquire)) return std::nullopt;
    const auto t0 = std::chrono::steady_clock::now();
    job_->thread.join();
    const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::unique_ptr<Job> job = std::move(job_);
    const size_t tokens = job->image.live.ids.size();
    std::string error = job->error;
    if (job->ok && commit(std::move(job->entry), job->tmp, error))
        std::fprintf(stderr, "strata serve: conversation cache disk: wrote %zu tokens in %.1f ms (in the background; "
                     "%.1f ms waited for it); entries=%zu bytes=%zu\n", tokens, job->ms, waited, entries_.size(), bytes());
    else
        std::fprintf(stderr, "strata serve: conversation cache disk: not written (%s)\n", error.c_str());
    return std::move(job->image);
}

bool ConversationDisk::load_file(const std::string& path, uint64_t size, SavedConversation& image,
                                 std::string& error) const {
    std::vector<char> buf(kIoBuffer);   // outlives the FILE that uses it
    File f{std::fopen(path.c_str(), "rb")};
    if (!f) { error = "cannot open"; return false; }
    std::setvbuf(f.get(), buf.data(), _IOFBF, buf.size());
    Reader r{f.get(), size};
    bool cvec = true;
    std::vector<int64_t> lens;
    std::vector<uint64_t> hashes;
    std::vector<std::vector<ConversationImageKey>> imgs;
    if (!read_header(r, fingerprint_, cvec, lens, hashes, imgs)) {
        drop_pages(f.get());
        error = "foreign or damaged header";
        return false;
    }
    r.summed = true;
    SavedConversation out;
    for (auto& x : out.geometry) r.pod(x);
    r.pod(out.layer_lo); r.pod(out.layer_hi);
    read_checkpoint(r, out.live);
    const uint64_t n_ck = r.count(64);
    if (r.ok) out.checkpoints.resize((size_t) n_ck);
    for (auto& c : out.checkpoints) read_checkpoint(r, c);
    const uint64_t n_kv = r.count(64);
    if (r.ok) out.kv.resize((size_t) n_kv);
    for (auto& k : out.kv) {
        r.pod(k.format); r.pod(k.cells); r.pod(k.heads); r.pod(k.head_dim); r.pod(k.page_size);
        r.pod(k.pooled_rows); r.pod(k.idx_dim);
        r.buffer(k.k); r.buffer(k.v); r.buffer(k.k_scale); r.buffer(k.v_scale); r.buffer(k.pooled);
    }
    r.summed = false;
    const uint64_t computed = r.sum.value();
    uint64_t stored = 0;
    r.pod(stored);
    out.cvec = cvec;
    bool ok = r.ok && stored == computed && std::fgetc(f.get()) == EOF && out.checkpoints.size() + 1 == lens.size();
    // the header's heads must describe the payload's tokens
    if (ok) {
        ok = (int64_t) out.live.ids.size() == lens[0] && prefix_hash(out.live.ids) == hashes[0];
        for (size_t i = 0; ok && i < out.checkpoints.size(); ++i)
            ok = (int64_t) out.checkpoints[i].ids.size() == lens[i + 1] &&
                 prefix_hash(out.checkpoints[i].ids) == hashes[i + 1];
    }
    drop_pages(f.get());
    if (!ok) {
        error = r.ok ? "checksum or layout mismatch" : "truncated or malformed";
        return false;
    }
    image = std::move(out);
    return true;
}

}  // namespace strata::core
