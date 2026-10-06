// src/core/glm_fetch.cpp - see include/strata/core/glm_fetch.hpp.
#include "strata/core/glm_fetch.hpp"

#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {
namespace sk = strata::kernels;

namespace {
int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v != nullptr && *v ? std::atoi(v) : fallback;
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

GlmFetch::~GlmFetch() {
    // the reader's threads write into the slots: every read finished or dropped before they go
    for (int s = 0; s < n_slots_ && src_ != nullptr; ++s)
        if (slots_[s].reading && !src_->read_cancel(slots_[s].t)) (void) src_->read_wait(slots_[s].t);
    // (a ticket that was never queued is done: read_wait returns at once)
    for (int s = 0; s < n_slots_; ++s) {
        if (slots_[s].host == nullptr) continue;
        if (slots_[s].pinned) cudaFreeHost(slots_[s].host);
        else std::free(slots_[s].host);
    }
    if (h_pids_) cudaFreeHost(h_pids_);
    if (d_plog_) cudaFree(d_plog_);
    if (d_pids_) cudaFree(d_pids_);
    if (d_pw_) cudaFree(d_pw_);
}

bool GlmFetch::init(const ModelGeometry& g, ExpertCache* cache, FileExpertSource* src, int max_t, uint64_t blob_bytes,
                    std::string& err) {
    g_ = g;
    cache_ = cache;
    src_ = src;
    max_t_ = max_t;
    k_ = (int) g.n_expert_used;
    blob_bytes_ = blob_bytes;
    active_ = false;
    if (env_int("STRATA_GLM_FETCH", 1) == 0 || src == nullptr || !src->gguf_mode()) return true;
    FileExpertSource::ReaderOptions o;
    o.threads = env_int("STRATA_FETCH_THREADS", o.threads);
    if (const char* v = std::getenv("STRATA_GLM_FETCH_PART"); v != nullptr && *v) o.part_bytes = std::strtoull(v, nullptr, 10);
    if (const char* v = std::getenv("STRATA_GLM_FILE_IO"); v != nullptr && *v) {
        const std::string io = v;
        if (io == "direct") o.io = FileExpertSource::ReadIo::Direct;
        else if (io == "pread") o.io = FileExpertSource::ReadIo::Pread;
        else if (io == "mmap") o.io = FileExpertSource::ReadIo::Mmap;
        else { err = "STRATA_GLM_FILE_IO must be direct, pread or mmap"; return false; }
    }
    if (!src->start_reader(o, err)) return false;
    cold_ = src->cold();
    const int cap = max_t * k_;
    n_slots_ = std::max(env_int("STRATA_GLM_FETCH_SLOTS", 256), 2 * cap + 16);
    slots_.reset(new Slot[(size_t) n_slots_]);
    look_ = std::max(0, std::min(2, env_int("STRATA_GLM_LOOKAHEAD", 1)));
    kp_ = std::max(1, std::min(std::min(32, (int) g.n_expert), env_int("STRATA_GLM_LOOKAHEAD_K", k_)));
    pred_.assign((size_t) g.n_layers, {});
    prev_.assign((size_t) g.n_layers, {});
    if (look_ != 0) {
        prs_.n_embd = (int) g.n_embd;
        prs_.n_expert = (int) g.n_expert;
        prs_.k = kp_;
        prs_.norm = g.expert_weights_norm;
        prs_.scale = g.expert_weights_scale;
        if (cudaMalloc((void**) &d_plog_, (size_t) max_t * g.n_expert * 4) != cudaSuccess ||
            cudaMalloc((void**) &d_pids_, (size_t) max_t * kp_ * 4) != cudaSuccess ||
            cudaMalloc((void**) &d_pw_, (size_t) max_t * kp_ * 4) != cudaSuccess ||
            cudaHostAlloc((void**) &h_pids_, (size_t) max_t * kp_ * 4, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            err = "glm fetch: the lookahead's buffers";
            return false;
        }
    }
    active_ = true;
    const auto& ro = src->reader_options();
    std::fprintf(stderr, "strata generate: glm5-next: file tier: %d reader threads, parts of %.2f MiB, %s reads; %d host "
                         "slots (%.2f GiB at most); lookahead %s (top-%d of the next layer's router)%s\n",
                 ro.threads, (double) ro.part_bytes / 1048576.0,
                 ro.io == FileExpertSource::ReadIo::Direct ? "O_DIRECT" : ro.io == FileExpertSource::ReadIo::Pread ? "pread" : "mmap",
                 n_slots_, (double) n_slots_ * (double) blob_bytes_ / 1073741824.0,
                 look_ == 0 ? "off" : look_ == 1 ? "on" : "predicting only", kp_,
                 cold_ ? "; STRATA_FILE_COLD: no blob served twice from a slot" : "");
    return true;
}

void GlmFetch::predict(const GlmLayerWeights& next, const float* x, int T, void* stream) {
    if (!lookahead() || next.router == nullptr || T < 1 || T > max_t_) return;
    cudaStream_t st = (cudaStream_t) stream;
    sk::router_f32_logits(next.router, x, d_plog_, prs_, T, st);
    sk::router_sigmoid_topk(d_plog_, next.exp_probs_b, d_pids_, d_pw_, prs_, T, st);
    cudaMemcpyAsync(h_pids_, d_pids_, (size_t) T * kp_ * 4, cudaMemcpyDeviceToHost, st);
    pending_layer_ = next.layer;
    pending_t_ = T;
}

void GlmFetch::free_slot(int s) {
    Slot& x = slots_[s];
    if (x.key >= 0) where_.erase(x.key);
    x.key = -1;
    x.reading = false;
    x.pin_call = -1;
    x.spec_call = -1;
}

int GlmFetch::alloc_slot() {
    int fresh = -1, lru = -1;
    for (int s = 0; s < n_slots_; ++s) {
        Slot& x = slots_[s];
        if (x.host == nullptr) {
            if (fresh < 0) fresh = s;
            continue;
        }
        if (x.key < 0) return s;
        if (x.pin_call >= call_ || x.spec_call >= call_) continue;
        if (x.reading) {
            if (!src_->read_done(x.t)) continue;
            x.reading = false;
        }
        if (lru < 0 || x.used < slots_[lru].used) lru = s;
    }
    if (fresh >= 0) {
        Slot& x = slots_[fresh];
        void* p = nullptr;
        if (cudaHostAlloc(&p, (size_t) blob_bytes_, cudaHostAllocDefault) == cudaSuccess && p != nullptr) {
            x.pinned = true;
        } else {
            cudaGetLastError();
            p = nullptr;
            if (posix_memalign(&p, 4096, (size_t) blob_bytes_) != 0) p = nullptr;
            x.pinned = false;
        }
        if (p != nullptr) {
            x.host = (uint8_t*) p;
            return fresh;
        }
    }
    if (lru >= 0) free_slot(lru);
    return lru;
}

int GlmFetch::issue(int64_t l, int32_t e, bool urgent, int64_t pin_call) {
    int s = alloc_slot();
    if (s < 0 && urgent) {
        // every slot pinned or reading: wait for the oldest unpinned read (a speculative one nobody needs)
        for (int i = 0; i < n_slots_ && s < 0; ++i) {
            Slot& x = slots_[i];
            if (x.host != nullptr && x.reading && x.pin_call < call_ && x.spec_call < call_) {
                (void) src_->read_wait(x.t);
                free_slot(i);
                s = i;
            }
        }
    }
    if (s < 0) return -1;
    Slot& x = slots_[s];
    x.key = l * g_.n_expert + e;
    where_[x.key] = s;
    x.pin_call = pin_call;
    x.spec_call = urgent ? -1 : pin_call;
    x.read_for = pin_call;
    x.used = ++stamp_;
    x.reading = true;
    src_->read_async(x.t, l, e, x.host, urgent);
    return s;
}

void GlmFetch::begin(int64_t l, const int32_t* experts, const uint8_t* file, int n) {
    ++call_;
    const int64_t NE = g_.n_expert;
    auto has = [](const std::vector<int32_t>& v, int32_t e) { return std::find(v.begin(), v.end(), e) != v.end(); };
    // ---- the predictors, scored against this call's experts
    std::vector<int32_t> actual(experts, experts + n), fmiss;
    for (int i = 0; i < n; ++i)
        if (file[i]) fmiss.push_back(experts[i]);
    st_.file_actual += (int64_t) fmiss.size();
    if (!pred_[(size_t) l].empty()) {
        const auto& p = pred_[(size_t) l];
        ++st_.pred_calls;
        st_.pred_actual += n;
        st_.pred_size += (int64_t) p.size();
        for (int32_t e : actual) st_.pred_hit += has(p, e);
        for (int32_t e : fmiss) st_.file_pred_hit += has(p, e);
        st_.file_pred_actual += (int64_t) fmiss.size();
        pred_[(size_t) l].clear();
    }
    if (!prev_[(size_t) l].empty()) {
        const auto& p = prev_[(size_t) l];
        ++st_.prev_calls;
        st_.prev_actual += n;
        st_.prev_size += (int64_t) p.size();
        for (int32_t e : actual) st_.prev_hit += has(p, e);
        for (int32_t e : fmiss) st_.file_prev_hit += has(p, e);
        st_.file_prev_actual += (int64_t) fmiss.size();
    }
    prev_[(size_t) l] = actual;
    // ---- the speculative reads meant for this call: needed -> urgent, else dropped when nothing was read yet
    for (int s : spec_slots_) {
        Slot& x = slots_[s];
        if (x.spec_call != call_) continue;   // freed or reused meanwhile
        x.spec_call = -1;
        const bool need = x.key / NE == l && has(fmiss, (int32_t) (x.key % NE));
        if (need) {
            ++st_.spec_used;
            if (x.reading && !src_->read_done(x.t)) {
                src_->read_promote(x.t);
                ++st_.promoted;
            }
            continue;
        }
        if (x.reading && src_->read_cancel(x.t)) {
            ++st_.spec_cancelled;
            free_slot(s);
        } else {
            ++st_.spec_unused;
            x.pin_call = -1;   // stays as a cached blob once read
        }
    }
    spec_slots_.clear();
    // ---- this layer's file-tier misses, all at once
    if (!fmiss.empty()) ++st_.calls;
    for (int32_t e : fmiss) {
        const int64_t key = l * NE + e;
        auto it = where_.find(key);
        if (it != where_.end()) {
            Slot& x = slots_[it->second];
            x.used = ++stamp_;
            x.pin_call = call_;
            if (x.read_for == call_) continue;                    // the lookahead's read for this call
            if (!cold_ && !x.t.failed.load()) {                   // a blob kept from an earlier call
                if (!x.reading) ++st_.slot_hits;
                else src_->read_promote(x.t);                     // an unused lookahead read, still queued
                continue;
            }
            if (x.reading) {                                      // cold: an earlier call's read still running
                (void) src_->read_wait(x.t);
                x.reading = false;
            }
            free_slot(it->second);
        }
        if (issue(l, e, true, call_) >= 0) ++st_.urgent;
    }
    // ---- the lookahead: the next layer's predicted file-tier experts, behind this layer's reads
    if (pending_layer_ >= 0) {
        const int64_t ln = pending_layer_;
        pending_layer_ = -1;
        std::vector<int32_t> p;
        for (int i = 0; i < pending_t_ * kp_; ++i) {
            const int32_t e = h_pids_[i];
            if (e >= 0 && e < NE && !has(p, e)) p.push_back(e);
        }
        if (look_ == 1) {
            const int budget = std::max(1, n_slots_ / 4);
            for (int32_t e : p) {
                if ((int) spec_slots_.size() >= budget) break;
                if (cache_ != nullptr && cache_->slot_of(ln, e) != kNotResident) continue;
                if (src_->has_resident(ln, e)) continue;
                const int64_t key = ln * NE + e;
                if (auto it = where_.find(key); it != where_.end()) {
                    Slot& x = slots_[it->second];
                    if (!cold_ || x.reading || x.read_for == call_) {   // kept or already coming: keep it
                        x.pin_call = std::max<int64_t>(x.pin_call, call_ + 1);
                        continue;
                    }
                    free_slot(it->second);
                }
                const int s = issue(ln, e, false, call_ + 1);
                if (s < 0) break;
                spec_slots_.push_back(s);
                ++st_.spec;
            }
        }
        pred_[(size_t) ln] = std::move(p);
    }
}

const uint8_t* GlmFetch::wait(int64_t l, int32_t e, std::string& err) {
    const int64_t key = l * g_.n_expert + e;
    auto it = where_.find(key);
    if (it == where_.end()) {   // no slot was free when begin() asked: read it now
        if (issue(l, e, true, call_) < 0) { err = "glm fetch: no host slot for a file-tier expert"; return nullptr; }
        ++st_.urgent;
        it = where_.find(key);
    }
    Slot& x = slots_[it->second];
    if (x.reading) {
        const auto t0 = std::chrono::steady_clock::now();
        (void) src_->read_wait(x.t);
        st_.ms_wait += ms_since(t0);
        x.reading = false;
    }
    if (x.t.failed.load()) {   // the reader failed: the mapped copy, as the old path
        if (!src_->copy_blob(l, e, x.host)) {
            err = "glm fetch: reading layer " + std::to_string(l) + " expert " + std::to_string(e) + " failed";
            free_slot(it->second);
            return nullptr;
        }
        x.t.failed.store(false);
    }
    ++st_.file;
    return x.host;
}

void GlmFetch::end() {
    // the layer's copies have landed: its slots are evictable from the next call on (pin_call < call_)
}

std::string GlmFetch::stats_line() const {
    if (!active_) return "glm fetch: off (the old path: one blob at a time through FileExpertSource::blob)";
    const auto r = src_->reader_stats();
    const auto& ro = src_->reader_options();
    auto pct = [](int64_t a, int64_t b) { return b > 0 ? 100.0 * (double) a / (double) b : 0.0; };
    char b[1536];
    std::snprintf(
        b, sizeof b,
        "glm fetch: reader %d threads x %.1f MiB parts (%s): %lld blobs, %.1f MB in %.1f ms busy = %.2f GB/s (%.1f ms "
        "of part reads summed over threads); the engine waited %.1f ms; %lld file-tier experts: %lld kept in a slot "
        "from an earlier call, %lld urgent reads (%lld promoted from the lookahead); lookahead: %lld reads, %lld used, "
        "%lld dropped unread, %lld read unused; predicted sets vs each call's experts (every tier): next-layer router "
        "%lld/%lld = %.1f%% (%.1f per call, %lld calls), previous token %lld/%lld = %.1f%% (%.1f per call); file-tier misses "
        "in those calls: next-layer router %lld/%lld = %.1f%%, previous token %lld/%lld = %.1f%%",
        ro.threads, (double) ro.part_bytes / 1048576.0,
        ro.io == FileExpertSource::ReadIo::Direct ? "O_DIRECT" : ro.io == FileExpertSource::ReadIo::Pread ? "pread" : "mmap",
        (long long) r.blobs, (double) r.bytes / 1e6, r.busy_ms, r.busy_ms > 0 ? (double) r.bytes / 1e6 / r.busy_ms : 0.0,
        r.part_ms, st_.ms_wait, (long long) st_.file, (long long) st_.slot_hits, (long long) st_.urgent,
        (long long) st_.promoted, (long long) st_.spec, (long long) st_.spec_used, (long long) st_.spec_cancelled,
        (long long) st_.spec_unused, (long long) st_.pred_hit, (long long) st_.pred_actual,
        pct(st_.pred_hit, st_.pred_actual), st_.pred_calls ? (double) st_.pred_size / (double) st_.pred_calls : 0.0,
        (long long) st_.pred_calls, (long long) st_.prev_hit, (long long) st_.prev_actual,
        pct(st_.prev_hit, st_.prev_actual), st_.prev_calls ? (double) st_.prev_size / (double) st_.prev_calls : 0.0,
        (long long) st_.file_pred_hit, (long long) st_.file_pred_actual, pct(st_.file_pred_hit, st_.file_pred_actual),
        (long long) st_.file_prev_hit, (long long) st_.file_prev_actual, pct(st_.file_prev_hit, st_.file_prev_actual));
    return b;
}

}  // namespace strata::core
