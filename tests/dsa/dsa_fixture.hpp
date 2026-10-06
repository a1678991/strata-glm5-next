// tests/dsa/dsa_fixture.hpp - device-side helpers shared by the DSA parity tests (weights upload, a cache, a case's
// random fixture). Header-only; host C++ calling the CUDA runtime (the HIP shim maps it on AMD).
#pragma once

#include "dsa_oracle.hpp"
#include "strata/kernels/dsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace dsa_test {

namespace sk = strata::kernels;

inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T>
T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, v.size() * sizeof(T) + 256), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <typename T>
std::vector<T> download(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

/// The weights on the device; freed by the destructor.
struct DeviceWeights {
    sk::DsaWeights w;
    std::vector<void*> owned;
    template <typename T>
    const T* put(const std::vector<T>& v) {
        T* p = upload(v);
        owned.push_back(p);
        return p;
    }
    explicit DeviceWeights(const HostWeights& h) {
        w.wq_a = put(h.wq_a);
        w.wq_b = put(h.wq_b);
        w.wkv_a = put(h.wkv_a);
        w.wk_b = put(h.wk_b);
        w.wv_b = put(h.wv_b);
        w.wo = put(h.wo);
        w.idx_q_b = put(h.idx_q_b);
        w.idx_k = put(h.idx_k);
        w.idx_gate = put(h.idx_gate);
        w.q_a_norm = put(h.q_a_norm);
        w.kv_a_norm = put(h.kv_a_norm);
        w.idx_k_norm = put(h.idx_k_norm);
        w.idx_k_norm_b = put(h.idx_k_norm_b);
        w.idx_ape = put(h.idx_ape);
        w.idx_proj = put(h.idx_proj);
    }
    ~DeviceWeights() {
        for (void* p : owned) (void) cudaFree(p);
    }
    DeviceWeights(const DeviceWeights&) = delete;
    DeviceWeights& operator=(const DeviceWeights&) = delete;
};

/// A fully resident cache of `max_cells` cells; freed by the destructor.
struct DeviceCache {
    sk::DsaKvCache c;
    void* base = nullptr;
    DeviceCache(const DsaShapes& s, int fmt, int64_t max_cells, cudaStream_t st) {
        const uint64_t bytes = sk::dsa_kv_bytes(s, fmt, max_cells, 0);
        ck(cudaMalloc(&base, bytes), "cache malloc");
        sk::dsa_kv_carve(base, s, fmt, max_cells, 0, c);
        sk::dsa_kv_reset(c, s, st);
    }
    ~DeviceCache() { (void) cudaFree(base); }
    DeviceCache(const DeviceCache&) = delete;
    DeviceCache& operator=(const DeviceCache&) = delete;
};

/// A case's random inputs: the cells before the window (as the projections would leave them) and the window's x.
struct CaseInputs {
    std::vector<float> old_latent, old_key, old_gate, x;
};
inline CaseInputs make_case(const DsaShapes& s, const HostWeights& w, int64_t p0, int64_t T, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    CaseInputs in;
    in.old_latent.resize((size_t) (p0 * s.kv_lora));
    in.old_key.resize((size_t) (p0 * s.idx_dim));
    in.old_gate.resize((size_t) (p0 * s.idx_dim));
    for (int64_t i = 0; i < p0; ++i) {
        for (int64_t d = 0; d < s.kv_lora; ++d) in.old_latent[i * s.kv_lora + d] = nd(rng) * w.kv_a_norm[d];
        for (int64_t d = 0; d < s.idx_dim; ++d) {
            in.old_key[i * s.idx_dim + d] = nd(rng) * w.idx_k_norm[d] + w.idx_k_norm_b[d];
            in.old_gate[i * s.idx_dim + d] = nd(rng);
        }
    }
    in.x.resize((size_t) (T * s.n_embd));
    for (float& v : in.x) v = nd(rng);
    return in;
}

/// The fixture's old cells into both caches, and their pools.
inline void fill_old(const DsaShapes& s, const HostWeights& w, const sk::DsaWeights& dw, const CaseInputs& in,
                     int64_t p0, sk::DsaKvCache& dc, HostCache& hc, cudaStream_t st, int n_threads) {
    if (p0 > 0) {
        float* lat = upload(in.old_latent);
        float* key = upload(in.old_key);
        float* gate = upload(in.old_gate);
        sk::dsa_kv_append_range(dc, s, 0, p0, lat, key, gate, st);
        sk::dsa_pool_range(dc, s, dw.idx_ape, 0, p0 / s.kpool, st);
        ck(cudaStreamSynchronize(st), "fill sync");
        (void) cudaFree(lat);
        (void) cudaFree(key);
        (void) cudaFree(gate);
        cache_append(hc, s, 0, p0, in.old_latent.data(), in.old_key.data(), in.old_gate.data());
        oracle_pool(hc, s, w.idx_ape, 0, p0 / s.kpool, n_threads);
    }
}

/// Selection lists -> the [T][cap] id rows the kernels read (ascending, unused tail of each row zero).
inline std::vector<int32_t> id_rows(const std::vector<std::vector<int32_t>>& sel, int64_t cap) {
    std::vector<int32_t> ids(sel.size() * (size_t) cap, 0);
    for (size_t t = 0; t < sel.size(); ++t)
        std::copy(sel[t].begin(), sel[t].end(), ids.begin() + (ptrdiff_t) (t * (size_t) cap));
    return ids;
}
/// Ascending live cells -> llama.cpp-style slots (live first, the rest dead), n_sel wide.
inline std::vector<std::vector<int32_t>> slots_of(const std::vector<std::vector<int32_t>>& sel, int64_t n_sel) {
    std::vector<std::vector<int32_t>> out(sel.size());
    for (size_t t = 0; t < sel.size(); ++t) {
        out[t] = sel[t];
        out[t].resize((size_t) std::max<int64_t>(n_sel, (int64_t) sel[t].size()), -1);
    }
    return out;
}
/// |a and b| / |b| for two ascending lists.
inline double overlap(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t i = 0, j = 0, both = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { ++both; ++i; ++j; }
        else if (a[i] < b[j]) ++i;
        else ++j;
    }
    return b.empty() ? 1.0 : (double) both / (double) b.size();
}

}  // namespace dsa_test
