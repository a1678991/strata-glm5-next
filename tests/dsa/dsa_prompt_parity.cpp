// tests/dsa/dsa_prompt_parity.cpp - the DSA layer's prompt half (port plan Stage 6) against ggml-cpu at llama.cpp
// f1cee99 (dsa_oracle.hpp: build_kpool_select / build_dsa_layer's ops for a batch of tokens).
//
// A prompt chunk of T tokens at positions p0 .. p0 + T - 1 after a random fixture of p0 cells, as GlmPrefill reads
// it: the chunk's latents and key | gate rows appended at once (dsa_kv_append_range), the pools it completes pooled
// (dsa_pool_range over [p0 / kpool, (p0 + T) / kpool)), the lightning-indexer scores of the queries past 2,051 cells
// only (dsa_index_scores on the chunk's sparse suffix), the selection of every query (dsa_select: the identity for
// the dense ones), and the attention of every query (glm::dsa_prompt_attend).  Checked, per case:
//   1. the key | gate rows bitwise and the pooled rows within one f16 ulp (dsa_indexer_parity's criterion);
//   2. the selection of every token: equal to the oracle's (llama.cpp's top_k over the masked pool scores, the pools
//      expanded, the tail appended), or a near tie (every swapped pool's oracle score within twice the measured score
//      error of the threshold), on the oracle's pooled rows so the scorer and selection are measured alone;
//   3. the attention output o_lat [T][64][512] over the oracle's selection against the oracle's gather path (G:974-990;
//      the scatter path llama.cpp takes for a prompt is the same softmax over the same cells), tolerance 1e-5 of max
//      |ref| for both latent formats (the oracle's get_rows hands the cached f16 or dequantized int8 values on as f32,
//      so both sides multiply the same values in f32, in another order); and against the decode kernel dsa_attend on
//      the same ids (1e-5: two f32 softmax schemes, chunked partials vs online).
// Cases (both latent formats): p0 0 / T 96 (dense from the start), p0 1950 / T 256 (the chunk crosses 2,051 cells),
// p0 3001 / T 200 (every query sparse, a pool straddling the chunk's start).  Prints the attention's time per token.
#include "dsa_fixture.hpp"
#include "dsa_oracle.hpp"

#include "strata/kernels/dsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/glm_prefill_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace dsa_test;
namespace sk = strata::kernels;
namespace pg = strata::prefill::glm;

namespace {

constexpr double kScoreTol = 1e-5, kAttnTol = 1e-5, kVsDecodeTol = 1e-5;

int64_t ulp_dist(uint16_t a, uint16_t b) {
    auto ord = [](uint16_t h) -> int64_t { return (h & 0x8000) ? -(int64_t) (h & 0x7fff) : (int64_t) h; };
    return std::llabs(ord(a) - ord(b));
}

bool run_case(const DsaShapes& s, const HostWeights& hw, const DeviceWeights& dw, int fmt, int64_t p0, int64_t T,
              uint64_t seed, cudaStream_t st, int threads) {
    const int64_t N = p0 + T, I = s.idx_dim, P = s.kpool, L = s.kv_lora, H = s.n_head;
    const int64_t cap = sk::dsa_sel_cap(s), max_pools = sk::dsa_max_pools(N, s), top = sk::dsa_top_pools(s);
    const CaseInputs in = make_case(s, hw, p0, T, seed * 7919ull + (uint64_t) p0);
    DeviceCache dc(s, fmt, N, st);
    HostCache hc;
    cache_init(hc, s, fmt, N);
    fill_old(s, hw, dw.w, in, p0, dc.c, hc, st, threads);
    StepOut A, C;
    oracle_project(hw, s, in.x, T, A, threads);
    cache_append(hc, s, p0, T, A.latent.data(), A.ik.data(), A.ig.data());
    oracle_pool(hc, s, hw.idx_ape, p0 / P, (p0 + T) / P, threads);
    oracle_select(hc, s, A, p0, T, C, threads);
    bool ok = true;

    // the chunk into the GPU cache as the prompt path writes it
    float* lat = upload(A.latent);
    float* ik = upload(A.ik);
    float* ig = upload(A.ig);
    sk::dsa_kv_append_range(dc.c, s, p0, T, lat, ik, ig, st);
    sk::dsa_pool_range(dc.c, s, dw.w.idx_ape, p0 / P, (p0 + T) / P, st);
    ck(cudaStreamSynchronize(st), "append");
    // 1. the rows
    int64_t kg_diff = 0, pooled_big = 0, pooled_ulp = 0;
    {
        const std::vector<uint16_t> gkg = download(dc.c.kg, (size_t) (N * 2 * I));
        for (size_t i = 0; i < gkg.size(); ++i) kg_diff += gkg[i] != hc.kg[i];
        const std::vector<uint16_t> gp = download(dc.c.pooled, (size_t) ((N / P) * I));
        double pmax = 0.0;
        for (uint16_t h : hc.pooled) pmax = std::max(pmax, (double) std::fabs(sk::f32_from_f16(h)));
        for (size_t i = 0; i < gp.size(); ++i) {
            const int64_t u = ulp_dist(gp[i], hc.pooled[i]);
            const double ad = std::fabs((double) sk::f32_from_f16(gp[i]) - (double) sk::f32_from_f16(hc.pooled[i]));
            pooled_ulp += u >= 1;
            pooled_big += u > 1 && ad > 1e-6 * pmax;
        }
        if (kg_diff || pooled_big) ok = false;
    }
    // 2. the selection, on the oracle's pooled rows
    ck(cudaMemcpy(dc.c.pooled, hc.pooled.data(), (size_t) ((N / P) * I) * 2, cudaMemcpyHostToDevice), "pooled");
    std::vector<int32_t> steps((size_t) T * sk::kStepCount);
    sk::dsa_step_fill(steps.data(), p0, T, s);
    int32_t* d_steps = upload(steps);
    float* iq = upload(A.iq);
    float* iw = upload(A.iw);
    float* scores = nullptr;
    int32_t* ids = nullptr;
    ck(cudaMalloc((void**) &scores, (size_t) T * max_pools * 4), "scores");
    ck(cudaMalloc((void**) &ids, (size_t) T * cap * 4), "ids");
    int64_t sparse0 = T;
    for (int64_t t = 0; t < T; ++t)
        if ((p0 + t + 1) / P > top) { sparse0 = t; break; }
    if (sparse0 < T)
        sk::dsa_index_scores(dc.c, s, iq + sparse0 * s.idx_n_head * I, iw + sparse0 * s.idx_n_head,
                             d_steps + sparse0 * sk::kStepCount, T - sparse0, max_pools, scores + sparse0 * max_pools, st);
    sk::dsa_select(scores, s, d_steps, T, max_pools, cap, ids, st);
    ck(cudaStreamSynchronize(st), "select");
    const std::vector<float> gs = download(scores, (size_t) (T * max_pools));
    double score_err = 0.0, score_max = 0.0;
    for (int64_t t = sparse0; t < T; ++t)
        for (int64_t j = 0; j < (p0 + t + 1) / P; ++j) {
            const double ref = C.scores[(size_t) (t * C.n_pool + j)];
            score_err = std::max(score_err, std::fabs((double) gs[(size_t) (t * max_pools + j)] - ref));
            score_max = std::max(score_max, std::fabs(ref));
        }
    const double score_rel = score_max > 0 ? score_err / score_max : score_err;
    if (!(score_rel <= kScoreTol)) ok = false;
    const std::vector<int32_t> gids = download(ids, (size_t) (T * cap));
    int64_t exact = 0, near = 0;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t pos = p0 + t, wdt = sk::dsa_sel_width(pos, s);
        const std::vector<int32_t> g(gids.begin() + t * cap, gids.begin() + t * cap + wdt);
        const std::vector<int32_t>& o = C.sel[(size_t) t];
        if (g == o) { ++exact; continue; }
        const int64_t n_pools_sel = std::min<int64_t>((pos + 1) / P, top);
        std::vector<int32_t> pg_, po, sym;
        for (int64_t i = 0; i < n_pools_sel; ++i) {
            pg_.push_back(g[(size_t) (i * P)] / (int32_t) P);
            po.push_back(o[(size_t) (i * P)] / (int32_t) P);
        }
        float thr = INFINITY;
        for (int32_t j : po) thr = std::min(thr, C.scores[(size_t) (t * C.n_pool + j)]);
        std::set_symmetric_difference(pg_.begin(), pg_.end(), po.begin(), po.end(), std::back_inserter(sym));
        bool tie = g.size() == o.size() &&
                   std::equal(g.end() - (wdt - n_pools_sel * P), g.end(), o.end() - (wdt - n_pools_sel * P));
        for (int32_t j : sym)
            if (std::fabs((double) C.scores[(size_t) (t * C.n_pool + j)] - thr) > 2.0 * score_err) tie = false;
        if (tie) ++near;
        else ok = false;
    }
    // 3. the attention over the oracle's selection
    const std::vector<int32_t> oids = id_rows(C.sel, cap);
    int32_t* d_oids = upload(oids);
    float* q_abs = upload(A.q_abs);
    float* o_lat = nullptr;
    ck(cudaMalloc((void**) &o_lat, (size_t) T * H * L * 4), "o_lat");
    pg::dsa_prompt_attend(dc.c, s, q_abs, d_oids, cap, d_steps, T, 1.0f / sqrtf((float) s.head_k), o_lat, st);
    ck(cudaStreamSynchronize(st), "attend");
    const std::vector<float> got = download(o_lat, (size_t) (T * H * L));
    // the oracle in groups of 16 queries (its gather path holds n_sel x T latent rows)
    std::vector<float> ref((size_t) (T * H * L));
    for (int64_t t0 = 0; t0 < T; t0 += 16) {
        const int64_t n = std::min<int64_t>(16, T - t0);
        StepOut a, r;
        a.q_abs.assign(A.q_abs.begin() + t0 * H * L, A.q_abs.begin() + (t0 + n) * H * L);
        std::vector<std::vector<int32_t>> sl(C.sel.begin() + t0, C.sel.begin() + t0 + n);
        oracle_attend(hw, hc, s, a, p0 + t0, n, sl, r, threads);
        std::copy(r.o_lat.begin(), r.o_lat.end(), ref.begin() + t0 * H * L);
    }
    const Err ea = compare(got.data(), ref.data(), (int64_t) got.size());
    const double attn_tol = kAttnTol;
    if (!(ea.rel <= attn_tol)) ok = false;
    // the decode kernel on the same ids, 8 queries at a time
    std::vector<float> dec((size_t) (T * H * L));
    {
        float* scr = nullptr;
        float* od = nullptr;
        ck(cudaMalloc((void**) &scr, sk::dsa_attend_scratch_floats(cap, 8, s) * 4), "attend scratch");
        ck(cudaMalloc((void**) &od, (size_t) 8 * H * L * 4), "od");
        for (int64_t t0 = 0; t0 < T; t0 += 8) {
            const int64_t n = std::min<int64_t>(8, T - t0);
            sk::dsa_attend(dc.c, s, q_abs + t0 * H * L, d_oids + t0 * cap, d_steps + t0 * sk::kStepCount, n, cap,
                           1.0f / sqrtf((float) s.head_k), scr, od, st);
            ck(cudaMemcpy(dec.data() + t0 * H * L, od, (size_t) n * H * L * 4, cudaMemcpyDeviceToHost), "od");
        }
        (void) cudaFree(scr);
        (void) cudaFree(od);
    }
    const Err ed = compare(got.data(), dec.data(), (int64_t) got.size());
    if (!(ed.rel <= kVsDecodeTol)) ok = false;
    std::printf("%-4s p0 %5lld T %3lld (sparse from row %lld) | key|gate diff %lld, pooled 1-ulp %lld >1-ulp %lld | "
                "scores %.1e | sel exact %lld/%lld near-tie %lld | attn vs oracle %.2e (tol %.0e) vs decode %.2e | %s\n",
                fmt == sk::kDsaLatInt8 ? "int8" : "f16", (long long) p0, (long long) T, (long long) sparse0,
                (long long) kg_diff, (long long) pooled_ulp, (long long) pooled_big, score_rel, (long long) exact,
                (long long) T, (long long) near, ea.rel, attn_tol, ed.rel, ok ? "ok" : "FAIL");
    std::fflush(stdout);
    for (void* p : {(void*) lat, (void*) ik, (void*) ig, (void*) d_steps, (void*) iq, (void*) iw, (void*) scores,
                    (void*) ids, (void*) d_oids, (void*) q_abs, (void*) o_lat})
        (void) cudaFree(p);
    return ok;
}

// the prompt attention's time: 256 queries over 2,051 selected cells each, 64 heads, latent 512
void timing(const DsaShapes& s, cudaStream_t st) {
    const int64_t N = 4096, T = 256, cap = sk::dsa_sel_cap(s), H = s.n_head, L = s.kv_lora;
    for (int fmt : {(int) sk::kDsaLatInt8, (int) sk::kDsaLatF16}) {
        DeviceCache dc(s, fmt, N, st);
        std::vector<float> lat((size_t) N * L), key((size_t) N * s.idx_dim, 0.f);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : lat) v = nd(rng);
        float* dl = upload(lat);
        float* dk = upload(key);
        sk::dsa_kv_append_range(dc.c, s, 0, N, dl, dk, dk, st);
        std::vector<int32_t> steps((size_t) T * sk::kStepCount), ids((size_t) T * cap);
        sk::dsa_step_fill(steps.data(), N - T, T, s);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t j = 0; j < cap; ++j) ids[(size_t) (t * cap + j)] = (int32_t) ((j * 7 + t) % (N - T));
        std::vector<float> q((size_t) T * H * L);
        for (float& v : q) v = nd(rng) * 0.05f;
        int32_t* dst = upload(steps);
        int32_t* di = upload(ids);
        float* dq = upload(q);
        float* o = nullptr;
        ck(cudaMalloc((void**) &o, (size_t) T * H * L * 4), "o");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float ms = 0;
        for (int r = 0; r < 3; ++r) {
            cudaEventRecord(e0, st);
            pg::dsa_prompt_attend(dc.c, s, dq, di, cap, dst, T, 0.0625f, o, st);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms, e0, e1);
        }
        std::printf("[timing] %s latent: %lld queries x 2,051 cells x 64 heads: %.2f ms (%.1f us per query)\n",
                    fmt == sk::kDsaLatInt8 ? "int8" : "f16", (long long) T, ms, 1000.0 * ms / T);
        for (void* p : {(void*) dl, (void*) dk, (void*) dst, (void*) di, (void*) dq, (void*) o}) (void) cudaFree(p);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int threads = 16;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: %s [--threads K]\n", argv[0]); return 2; }
    }
    const DsaShapes s = sk::dsa_glm5_shapes();
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const HostWeights hw = make_weights(s, 11);
    const DeviceWeights dw(hw);
    bool ok = true;
    const auto t0 = std::chrono::steady_clock::now();
    for (int fmt : {(int) sk::kDsaLatInt8, (int) sk::kDsaLatF16})
        for (const auto& c : {std::pair<int64_t, int64_t>{0, 96}, {1950, 256}, {3001, 200}})
            ok &= run_case(s, hw, dw, fmt, c.first, c.second, 11, st, threads);
    timing(s, st);
    std::printf("dsa_prompt_parity: %s (%.0f s)\n", ok ? "PASS" : "FAIL",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
