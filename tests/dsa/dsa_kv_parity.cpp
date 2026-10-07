// tests/dsa/dsa_kv_parity.cpp - the DSA layer's latent KV cache (dsa.hpp, mla_kv.cu) against a host reference.
// GPU, synthetic, no model, no oracle library:
//   1. int8 and f16: dsa_kv_append_range and dsa_kv_append (positions from step records) through a PERMUTED page
//      table; every slot's codes, scales, f16 values and key | gate rows BITWISE against the host rule (kv_q8.hpp's
//      int8 scheme, round-to-nearest-even f16), and dsa_kv_read_page's dequantized page equal to the host's;
//   2. streaming (--kv-resident): a cache with fewer slots than pages and a pinned, device-mapped host copy, filled
//      while nothing is resident: the pooling (which then reads the host copy) and the page read equal a fully
//      resident cache's bitwise; rounds of 8 tokens with random selections through dsa_kv_stream_resolve (kv_stream.cu
//      unchanged) and dsa_attend give the resident cache's attention output bitwise, with misses and evictions;
//   3. the bytes per cell and layer.
#include "strata/kernels/dsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace sk = strata::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, v.size() * sizeof(T) + 256), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <typename T> std::vector<T> download(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
void expect(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the engine's int8 latent rule on the host
void quant_row(const float* x, int n, int8_t* q, uint16_t* sc) {
    for (int g = 0; g < n / sk::DSA_LAT_GROUP; ++g) {
        float amax = 0.0f;
        for (int j = 0; j < sk::DSA_LAT_GROUP; ++j) amax = std::fmax(amax, std::fabs(x[g * sk::DSA_LAT_GROUP + j]));
        const uint16_t b = sk::f16_from_f32(amax / 127.0f);
        const float sf = sk::f32_from_f16(b);
        sc[g] = b;
        for (int j = 0; j < sk::DSA_LAT_GROUP; ++j) {
            int v = 0;
            if (sf > 0.0f) {
                v = (int) std::nearbyint(x[g * sk::DSA_LAT_GROUP + j] / sf);
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
            }
            q[g * sk::DSA_LAT_GROUP + j] = (int8_t) v;
        }
    }
}

struct Rows {
    std::vector<float> lat, key, gate;
};
Rows random_rows(const sk::DsaShapes& s, int64_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Rows r;
    r.lat.resize((size_t) (n * s.kv_lora));
    r.key.resize((size_t) (n * s.idx_dim));
    r.gate.resize((size_t) (n * s.idx_dim));
    for (float& v : r.lat) v = nd(rng) * 3.0f;
    for (float& v : r.key) v = nd(rng);
    for (float& v : r.gate) v = nd(rng);
    // a few groups of exact zeros (scale 0) and one huge value (f16 scale range)
    for (int d = 0; d < sk::DSA_LAT_GROUP; ++d) r.lat[(size_t) d] = 0.0f;
    r.lat[(size_t) (s.kv_lora + 5)] = 3.0e4f;
    return r;
}

// 1. one format through a permuted page table
void layout_case(const sk::DsaShapes& s, int fmt, cudaStream_t st) {
    std::printf("%s latent, permuted page table:\n", fmt == sk::kDsaLatInt8 ? "int8" : "f16");
    const int64_t N = 1000, P = s.kpool, L = s.kv_lora, I = s.idx_dim, G = L / sk::DSA_LAT_GROUP;
    sk::DsaKvCache c;
    void* base = nullptr;
    ck(cudaMalloc(&base, sk::dsa_kv_bytes(s, fmt, N, 0)), "cache");
    sk::dsa_kv_carve(base, s, fmt, N, 0, c);
    sk::dsa_kv_reset(c, s, st);
    std::vector<int32_t> table((size_t) c.n_pages);
    for (int64_t i = 0; i < c.n_pages; ++i) table[(size_t) i] = (int32_t) ((i * 97 + 13) % c.n_pages);   // 97 coprime
    ck(cudaMemcpy(c.page_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "table");
    const Rows r = random_rows(s, N, 5 + fmt);
    // cells 0..N-9 by range, the last 8 by step records (a verify window)
    float* lat = upload(r.lat);
    float* key = upload(r.key);
    float* gate = upload(r.gate);
    sk::dsa_kv_append_range(c, s, 0, N - 8, lat, key, gate, st);
    std::vector<int32_t> steps(8 * sk::kStepCount);
    sk::dsa_step_fill(steps.data(), N - 8, 8, s);
    int32_t* d_steps = upload(steps);
    sk::dsa_kv_append(c, s, d_steps, 8, lat + (N - 8) * L, key + (N - 8) * I, gate + (N - 8) * I, st);
    ck(cudaStreamSynchronize(st), "append");
    // slot contents bitwise
    const int64_t cells = c.n_slots * P;
    bool codes_ok = true, kg_ok = true;
    std::vector<int8_t> hq((size_t) L);
    std::vector<uint16_t> hs((size_t) G);
    std::vector<int8_t> gq;
    std::vector<uint16_t> gsc, gh;
    if (fmt == sk::kDsaLatInt8) {
        gq = download(c.lat_q, (size_t) (cells * L));
        gsc = download(c.lat_scale, (size_t) (cells * G));
    } else {
        gh = download(c.lat_f16, (size_t) (cells * L));
    }
    const std::vector<uint16_t> gkg = download(c.kg, (size_t) (cells * 2 * I));
    for (int64_t cell = 0; cell < N; ++cell) {
        const int64_t row = (int64_t) table[(size_t) (cell / P)] * P + cell % P;
        const float* x = r.lat.data() + cell * L;
        if (fmt == sk::kDsaLatInt8) {
            quant_row(x, (int) L, hq.data(), hs.data());
            codes_ok = codes_ok && std::memcmp(hq.data(), gq.data() + row * L, (size_t) L) == 0 &&
                       std::memcmp(hs.data(), gsc.data() + row * G, (size_t) G * 2) == 0;
        } else {
            for (int64_t d = 0; d < L; ++d) codes_ok = codes_ok && gh[(size_t) (row * L + d)] == sk::f16_from_f32(x[d]);
        }
        for (int64_t d = 0; d < I; ++d) {
            kg_ok = kg_ok && gkg[(size_t) (row * 2 * I + d)] == sk::f16_from_f32(r.key[(size_t) (cell * I + d)]);
            kg_ok = kg_ok && gkg[(size_t) (row * 2 * I + I + d)] == sk::f16_from_f32(r.gate[(size_t) (cell * I + d)]);
        }
    }
    expect(codes_ok, fmt == sk::kDsaLatInt8 ? "int8 codes and f16 scales of 1,000 cells bitwise (zero group, 3e4 value)"
                                            : "f16 latent of 1,000 cells bitwise");
    expect(kg_ok, "key | gate rows bitwise");
    // the page read
    float* d_lat = nullptr;
    float* d_kg = nullptr;
    ck(cudaMalloc((void**) &d_lat, (size_t) (P * L) * 4), "m");
    ck(cudaMalloc((void**) &d_kg, (size_t) (P * 2 * I) * 4), "m");
    bool page_ok = true;
    for (int64_t page : {(int64_t) 0, (int64_t) 1, (int64_t) 123, c.n_pages - 1}) {
        sk::dsa_kv_read_page(c, s, page, d_lat, d_kg, st);
        const std::vector<float> gl = download(d_lat, (size_t) (P * L));
        const std::vector<float> gk = download(d_kg, (size_t) (P * 2 * I));
        for (int64_t k = 0; k < P; ++k) {
            const int64_t cell = page * P + k;
            const float* x = r.lat.data() + cell * L;
            if (fmt == sk::kDsaLatInt8) quant_row(x, (int) L, hq.data(), hs.data());
            for (int64_t d = 0; d < L; ++d) {
                const float want = fmt == sk::kDsaLatInt8 ? (float) hq[(size_t) d] * sk::f32_from_f16(hs[(size_t) (d / sk::DSA_LAT_GROUP)])
                                                          : sk::f32_from_f16(sk::f16_from_f32(x[d]));
                page_ok = page_ok && gl[(size_t) (k * L + d)] == want;
            }
            for (int64_t d = 0; d < I; ++d)
                page_ok = page_ok && gk[(size_t) (k * 2 * I + d)] == sk::f32_from_f16(sk::f16_from_f32(r.key[(size_t) (cell * I + d)]));
        }
    }
    expect(page_ok, "dsa_kv_read_page: the dequantized pages 0, 1, 123 and the last");
    for (void* p : {(void*) lat, (void*) key, (void*) gate, (void*) d_steps, (void*) d_lat, (void*) d_kg, base})
        (void) cudaFree(p);
}

// 2. streaming
void streaming_case(const sk::DsaShapes& s, cudaStream_t st) {
    std::printf("streaming (int8, 32,768 cells, 4,200 of 8,192 pages resident):\n");
    const int64_t N = 32768, P = s.kpool, L = s.kv_lora, H = s.n_head, T = 8, n_slots = 4200;
    const int64_t cap = sk::dsa_sel_cap(s);
    // the streamed cache and its host copy; the resident reference
    sk::DsaKvCache cs, cr;
    void *bs = nullptr, *br = nullptr;
    ck(cudaMalloc(&bs, sk::dsa_kv_bytes(s, sk::kDsaLatInt8, N, n_slots)), "streamed");
    sk::dsa_kv_carve(bs, s, sk::kDsaLatInt8, N, n_slots, cs);
    void* hhost = nullptr;
    void* hdev = nullptr;
    const uint64_t hbytes = sk::dsa_kv_host_bytes(s, sk::kDsaLatInt8, N);
    ck(cudaHostAlloc(&hhost, hbytes, cudaHostAllocMapped | cudaHostAllocPortable), "pin");
    ck(cudaHostGetDevicePointer(&hdev, hhost, 0), "mapped");
    std::memset(hhost, 0, hbytes);
    sk::dsa_kv_host_carve(hdev, s, sk::kDsaLatInt8, N, cs);
    sk::dsa_kv_reset(cs, s, st);
    ck(cudaMalloc(&br, sk::dsa_kv_bytes(s, sk::kDsaLatInt8, N, 0)), "resident");
    sk::dsa_kv_carve(br, s, sk::kDsaLatInt8, N, 0, cr);
    sk::dsa_kv_reset(cr, s, st);
    expect(cs.n_slots == n_slots && cs.n_pages == N / P && cs.map.slot_block != nullptr && cr.n_slots == cr.n_pages,
           "carve: 4,200 slots of 8,192 pages with a residency map; the reference fully resident");
    const Rows r = random_rows(s, N, 99);
    float* lat = upload(r.lat);
    float* key = upload(r.key);
    float* gate = upload(r.gate);
    std::vector<float> ape((size_t) (P * s.idx_dim));
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    for (float& v : ape) v = nd(rng);
    float* d_ape = upload(ape);
    sk::dsa_kv_append_range(cs, s, 0, N, lat, key, gate, st);   // nothing resident: the host copy only
    sk::dsa_kv_append_range(cr, s, 0, N, lat, key, gate, st);
    sk::dsa_pool_range(cs, s, d_ape, 0, N / P, st);              // reads the host copy's key | gate rows
    sk::dsa_pool_range(cr, s, d_ape, 0, N / P, st);
    ck(cudaStreamSynchronize(st), "fill");
    expect(download(cs.pooled, (size_t) (N / P * s.idx_dim)) == download(cr.pooled, (size_t) (N / P * s.idx_dim)),
           "pooled rows from the host copy == the resident cache's, bitwise");
    float *d_lat = nullptr, *d_lat2 = nullptr;
    ck(cudaMalloc((void**) &d_lat, (size_t) (P * L) * 4), "m");
    ck(cudaMalloc((void**) &d_lat2, (size_t) (P * L) * 4), "m");
    sk::dsa_kv_read_page(cs, s, 4321, d_lat, nullptr, st);
    sk::dsa_kv_read_page(cr, s, 4321, d_lat2, nullptr, st);
    expect(download(d_lat, (size_t) (P * L)) == download(d_lat2, (size_t) (P * L)),
           "a non-resident page read from the host copy == the resident page");
    // rounds of T tokens at the end of the context, each with random selections over all of it
    std::vector<int32_t> steps((size_t) (T * sk::kStepCount));
    sk::dsa_step_fill(steps.data(), N - T, T, s);
    int32_t* d_steps = upload(steps);
    int32_t* d_ids = nullptr;
    float *d_q = nullptr, *d_os = nullptr, *d_or = nullptr, *d_scr = nullptr;
    ck(cudaMalloc((void**) &d_ids, (size_t) (T * cap) * 4), "ids");
    ck(cudaMalloc((void**) &d_q, (size_t) (T * H * L) * 4), "q");
    ck(cudaMalloc((void**) &d_os, (size_t) (T * H * L) * 4), "o");
    ck(cudaMalloc((void**) &d_or, (size_t) (T * H * L) * 4), "o");
    ck(cudaMalloc((void**) &d_scr, sk::dsa_attend_scratch_floats(cap, T, s) * 4), "scratch");
    bool same = true;
    for (int round = 0; round < 6; ++round) {
        std::vector<float> q((size_t) (T * H * L));
        std::normal_distribution<float> qn(0.0f, 0.05f);
        for (float& v : q) v = qn(rng);
        ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");
        std::vector<int32_t> ids((size_t) (T * cap), 0);
        for (int64_t t = 0; t < T; ++t) {
            const int64_t pos = N - T + t, nv = (pos + 1) / P, n_tail = (pos + 1) % P;
            std::vector<int32_t> pools((size_t) nv);
            std::iota(pools.begin(), pools.end(), 0);
            std::shuffle(pools.begin(), pools.end(), rng);
            pools.resize((size_t) sk::dsa_top_pools(s));
            std::sort(pools.begin(), pools.end());
            size_t k = 0;
            for (int32_t j : pools)
                for (int64_t m = 0; m < P; ++m) ids[(size_t) (t * cap) + k++] = (int32_t) (j * P + m);
            for (int64_t m = n_tail - 1; m >= 0; --m) ids[(size_t) (t * cap) + k++] = (int32_t) (pos - m);
        }
        ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
        sk::dsa_kv_stream_resolve(cs, s, d_ids, d_steps, T, cap, st);
        sk::dsa_attend(cs, s, d_q, d_ids, d_steps, T, cap, 0.0625f, d_scr, d_os, st);
        sk::dsa_attend(cr, s, d_q, d_ids, d_steps, T, cap, 0.0625f, d_scr, d_or, st);
        ck(cudaStreamSynchronize(st), "round");
        same = same && download(d_os, q.size()) == download(d_or, q.size());
    }
    const sk::KvStreamCounters k = sk::kv_stream_counters(cs.map);
    std::printf("  stream counters: %llu calls, %llu lookups, %llu misses, overflow %d\n", (unsigned long long) k.calls,
                (unsigned long long) k.lookups, (unsigned long long) k.misses, (int) k.overflow);
    expect(same, "6 rounds x 8 tokens x 2,051 cells: streamed attention == resident, bitwise");
    expect(k.calls == 6 && k.misses > (uint64_t) n_slots && !k.overflow,
           "the rounds missed more pages than the slots hold (evictions), no overflow");
    sk::dsa_kv_read_page(cs, s, 8191, d_lat, nullptr, st);   // the tail's page: resident now
    sk::dsa_kv_read_page(cr, s, 8191, d_lat2, nullptr, st);
    int32_t slot = -1;
    ck(cudaMemcpy(&slot, cs.page_table + 8191, 4, cudaMemcpyDeviceToHost), "slot");
    expect(slot >= 0 && download(d_lat, (size_t) (P * L)) == download(d_lat2, (size_t) (P * L)),
           "the last page, made resident by the resolve, reads as the reference's");
    for (void* p : {(void*) lat, (void*) key, (void*) gate, (void*) d_ape, (void*) d_lat, (void*) d_lat2, (void*) d_steps,
                    (void*) d_ids, (void*) d_q, (void*) d_os, (void*) d_or, (void*) d_scr, bs, br})
        (void) cudaFree(p);
    (void) cudaFreeHost(hhost);
}

}  // namespace

int main() {
    const sk::DsaShapes s = sk::dsa_glm5_shapes();
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    layout_case(s, sk::kDsaLatInt8, st);
    layout_case(s, sk::kDsaLatF16, st);
    streaming_case(s, st);
    std::printf("bytes per cell and DSA layer (GLM-5.3-Flash geometry; x 11 DSA layers):\n");
    for (int fmt : {(int) sk::kDsaLatInt8, (int) sk::kDsaLatF16}) {
        const uint64_t b = sk::dsa_kv_cell_bytes(s, fmt);
        std::printf("  %-5s page %llu B, cell %llu B (x 11 = %llu B/token)\n", fmt == sk::kDsaLatInt8 ? "int8" : "f16",
                    (unsigned long long) sk::dsa_kv_page_bytes(s, fmt), (unsigned long long) b,
                    (unsigned long long) (11 * b));
    }
    std::printf("  llama.cpp f16: 1792 B (x 11 = 19712 B/token)\n");
    expect(sk::dsa_kv_cell_bytes(s, sk::kDsaLatInt8) == 1120 && sk::dsa_kv_cell_bytes(s, sk::kDsaLatF16) == 1600,
           "int8 1,120 B and f16 1,600 B per cell and layer");
    (void) cudaStreamDestroy(st);
    std::printf("%s\n", g_fail ? "FAIL" : "all ok");
    return g_fail ? 1 : 0;
}
