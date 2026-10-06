// tests/dsa/dsa_indexer_parity.cpp - the DSA k-pool indexer (append, pooling, lightning scores, top-k selection)
// against ggml-cpu at llama.cpp f1cee99 (`build_kpool_select`, src/models/glm5-next.cpp:766-919, dsa_oracle.hpp).
//
// Per seed, cache size N and step width T (positions N-T .. N-1 after a random fixture of N-T cells):
//   1. the key | gate rows (f16, bitwise) and the pooled rows (GPU dsa_pool_range / dsa_pool_step vs ggml's
//      pooling) of every cell and pool;
//   2. ISOLATED: the oracle's iq, w, key, gate and latent into the GPU kernels (dsa_kv_append, dsa_pool_step,
//      dsa_index_scores, dsa_select): the scores of every visible pool, and the selected cells per token, which
//      must be the oracle's exactly unless every swapped pool's oracle score lies within twice the measured score
//      error of the oracle's selection threshold (a near-tie: counted and reported);
//   3. the indexer's projections on the oracle's inputs (idx_k, the key LayerNorm, idx_gate, idx_proj, idx_q_b),
//      with dsa_mla_parity's flip-aware tolerance for the quantized ones;
//   4. END TO END: dsa_decode from x; its selection's overlap with the oracle's (reported; exact while dense).
//   --seeds K (default 3)   --quick (N in {64, 2052})   --threads K (oracle, default 8)
#include "dsa_fixture.hpp"
#include "dsa_oracle.hpp"

#include "strata/kernels/dsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace dsa_test;
namespace sk = strata::kernels;

namespace {

// Scores: 128-term f32 dots and a 32-term head sum in another order: 1e-5 of max |score|.
// Norm: the same double sums as ggml: 1e-6. Quantized projections: 1e-5 + the flip bound (dsa_mla_parity).
// Pooled rows are f16: the GPU's expf and glibc's may differ in the last f32 bit, which can move an f16 rounding by
// one ulp; one ulp is accepted (and counted), more is a failure.
constexpr double kScoreTol = 1e-5, kNormTol = 1e-6, kF32Tol = 1e-5;
constexpr int kProjN = 5;
constexpr const char* kProj[kProjN] = {"idx_k", "k_norm", "idx_gate", "idx_proj", "idx_q_b"};

struct Agg {
    int cases = 0, failed = 0;
    int64_t kg_diff = 0, kg_total = 0, pooled_ulp = 0, pooled_big = 0, pooled_total = 0;
    double pooled_max_abs = 0.0;
    double worst_score = 0.0, worst_proj[kProjN] = {}, worst_ratio[kProjN] = {}, worst_noflip[kProjN] = {};
    int64_t proj_flips[kProjN] = {}, proj_off_rule[kProjN] = {}, proj_bad[kProjN] = {};
    int64_t tokens = 0, exact_tokens = 0, near_tie_tokens = 0;
    double min_overlap_iso = 1.0, min_overlap_e2e = 1.0;
    int64_t e2e_tokens = 0, e2e_exact = 0;
};

int64_t ulp_dist(uint16_t a, uint16_t b) {
    auto ord = [](uint16_t h) -> int64_t { return (h & 0x8000) ? -(int64_t) (h & 0x7fff) : (int64_t) h; };
    return std::llabs(ord(a) - ord(b));
}

void run_case(const DsaShapes& s, const HostWeights& hw, const DeviceWeights& dw, int64_t N, int64_t T, uint64_t seed,
              const sk::DsaScratch& sc, int32_t* d_steps, float* d_x, float* d_out, float* d_a, float* d_b, void* d_xq,
              cudaStream_t st, int threads, Agg& agg) {
    const int64_t p0 = N - T, I = s.idx_dim, P = s.kpool;
    const CaseInputs in = make_case(s, hw, p0, T, seed * 1000003ull + (uint64_t) N * 31 + (uint64_t) T);
    DeviceCache dc(s, sk::kDsaLatInt8, N, st);
    HostCache hc;
    cache_init(hc, s, sk::kDsaLatInt8, N);
    fill_old(s, hw, dw.w, in, p0, dc.c, hc, st, threads);
    bool ok = true;

    StepOut A, C;
    oracle_project(hw, s, in.x, T, A, threads);
    cache_append(hc, s, p0, T, A.latent.data(), A.ik.data(), A.ig.data());
    oracle_pool(hc, s, hw.idx_ape, p0 / P, (p0 + T) / P, threads);
    oracle_select(hc, s, A, p0, T, C, threads);

    // 1. the rows: key | gate bitwise for every cell; pooled within one f16 ulp, or within 1e-6 of the largest
    //    |pooled| where the members cancel (a near-zero value has no meaningful ulp distance)
    auto check_rows = [&] {
        const std::vector<uint16_t> gkg = download(dc.c.kg, (size_t) (N * 2 * I));
        int64_t d = 0;
        for (size_t i = 0; i < gkg.size(); ++i) d += gkg[i] != hc.kg[i];
        agg.kg_diff += d;
        agg.kg_total += (int64_t) gkg.size();
        if (d) ok = false;
        const std::vector<uint16_t> gp = download(dc.c.pooled, (size_t) ((N / P) * I));
        double pmax = 0.0;
        for (uint16_t h : hc.pooled) pmax = std::max(pmax, (double) std::fabs(sk::f32_from_f16(h)));
        int64_t big = 0;
        for (size_t i = 0; i < gp.size(); ++i) {
            const int64_t u = ulp_dist(gp[i], hc.pooled[i]);
            const double ad = std::fabs((double) sk::f32_from_f16(gp[i]) - (double) sk::f32_from_f16(hc.pooled[i]));
            agg.pooled_ulp += u >= 1;
            agg.pooled_max_abs = std::max(agg.pooled_max_abs, pmax > 0 ? ad / pmax : ad);
            big += u > 1 && ad > 1e-6 * pmax;
        }
        agg.pooled_big += big;
        agg.pooled_total += (int64_t) gp.size();
        if (big) ok = false;
    };

    // 2. isolated: the oracle's indexer inputs through the GPU kernels
    std::vector<int32_t> steps((size_t) T * sk::kStepCount);
    sk::dsa_step_fill(steps.data(), p0, T, s);
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
    {
        float* lat = upload(A.latent);
        float* ik = upload(A.ik);
        float* ig = upload(A.ig);
        float* iq = upload(A.iq);
        float* iw = upload(A.iw);
        sk::dsa_kv_append(dc.c, s, d_steps, T, lat, ik, ig, st);
        sk::dsa_pool_step(dc.c, s, dw.w.idx_ape, d_steps, T, st);
        ck(cudaStreamSynchronize(st), "isolated");
        check_rows();
        // the scorer and the selection on the oracle's pooled rows, so that each is measured alone
        ck(cudaMemcpy(dc.c.pooled, hc.pooled.data(), (size_t) ((N / P) * I) * 2, cudaMemcpyHostToDevice), "pooled");
        sk::dsa_index_scores(dc.c, s, iq, iw, d_steps, T, sc.max_pools, sc.scores, st);
        sk::dsa_select(sc.scores, s, d_steps, T, sc.max_pools, sc.cap, sc.ids, st);
        ck(cudaStreamSynchronize(st), "isolated");
        for (float* p : {lat, ik, ig, iq, iw}) (void) cudaFree(p);
    }
    // the scores of every visible pool
    const std::vector<float> gs = download(sc.scores, (size_t) (T * sc.max_pools));
    double score_err = 0.0, score_max = 0.0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < (p0 + t + 1) / P; ++j) {
            const double ref = C.scores[(size_t) (t * C.n_pool + j)];
            score_err = std::max(score_err, std::fabs((double) gs[(size_t) (t * sc.max_pools + j)] - ref));
            score_max = std::max(score_max, std::fabs(ref));
        }
    const double score_rel = score_max > 0 ? score_err / score_max : score_err;
    agg.worst_score = std::max(agg.worst_score, score_rel);
    if (!(score_rel <= kScoreTol)) ok = false;
    // the selection: both lists are ascending, the selected pools' cells first, then the tail
    const std::vector<int32_t> gids = download(sc.ids, (size_t) (T * sc.cap));
    int exact = 0, near = 0;
    double ov_iso = 1.0;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t pos = p0 + t, wdt = sk::dsa_sel_width(pos, s);
        const std::vector<int32_t> g(gids.begin() + t * sc.cap, gids.begin() + t * sc.cap + wdt);
        const std::vector<int32_t>& o = C.sel[(size_t) t];
        ov_iso = std::min(ov_iso, overlap(g, o));
        if (g == o) { ++exact; continue; }
        const int64_t n_pools_sel = std::min<int64_t>((pos + 1) / P, sk::dsa_top_pools(s));
        std::vector<int32_t> pg, po, sym;
        for (int64_t i = 0; i < n_pools_sel; ++i) {
            pg.push_back(g[(size_t) (i * P)] / (int32_t) P);
            po.push_back(o[(size_t) (i * P)] / (int32_t) P);
        }
        float thr = INFINITY;
        for (int32_t j : po) thr = std::min(thr, C.scores[(size_t) (t * C.n_pool + j)]);
        std::set_symmetric_difference(pg.begin(), pg.end(), po.begin(), po.end(), std::back_inserter(sym));
        bool tie = std::equal(g.end() - (wdt - n_pools_sel * P), g.end(), o.end() - (wdt - n_pools_sel * P));
        for (int32_t j : sym)
            if (std::fabs((double) C.scores[(size_t) (t * C.n_pool + j)] - thr) > 2.0 * score_err) tie = false;
        if (tie) ++near;
        else ok = false;
    }
    agg.tokens += T;
    agg.exact_tokens += exact;
    agg.near_tie_tokens += near;
    agg.min_overlap_iso = std::min(agg.min_overlap_iso, ov_iso);

    // 3. the projections on the oracle's inputs
    double perr[kProjN], ptol[kProjN];
    int64_t pfl[kProjN] = {}, poff[kProjN] = {}, pbad[kProjN] = {};
    auto proj = [&](int i, const std::vector<float>& x, int64_t cols, int64_t n_in, double max_w,
                    const std::vector<float>& ref, const std::function<void()>& run, double fixed_tol) {
        ck(cudaMemcpy(d_a, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "proj in");
        run();
        ck(cudaStreamSynchronize(st), "proj");
        const std::vector<float> got = download(d_b, ref.size());
        const Err e = compare(got.data(), ref.data(), (int64_t) ref.size());
        perr[i] = e.rel;
        if (fixed_tol > 0) {
            ptol[i] = fixed_tol;
            return;
        }
        // the flip bound from the GPU's own Q8_1 blocks of x (dsa_mla_parity, q8_1_flip_bound)
        const std::vector<uint8_t> xq = download(static_cast<const uint8_t*>(d_xq), (size_t) (cols * (n_in / 32) * 36));
        const FlipBound fb = q8_1_flip_bound(x.data(), xq.data(), cols, n_in);
        pfl[i] = fb.flips;
        poff[i] = fb.off_rule;
        pbad[i] = fb.bad;
        ptol[i] = kF32Tol + 1.5 * fb.steps * max_w / e.ref_max;
    };
    const int E = (int) s.n_embd, n = (int) T;
    const auto& w = dw.w;
    proj(0, in.x, T, E, q8_0_max_abs(hw.idx_k), A.ik_raw, [&] {
        sk::native_quantize_q8_1(d_a, d_xq, E, n, st);
        sk::native_mmvq(8, w.idx_k, d_xq, d_b, E, (int) I, n, st);
    }, 0.0);
    proj(1, A.ik_raw, T, I, 0.0, A.ik, [&] {
        sk::dsa_layer_norm(d_a, w.idx_k_norm, w.idx_k_norm_b, s.norm_eps, I, T, d_b, st);
    }, kNormTol);
    proj(2, in.x, T, E, q8_0_max_abs(hw.idx_gate), A.ig, [&] {
        sk::native_quantize_q8_1(d_a, d_xq, E, n, st);
        sk::native_mmvq(8, w.idx_gate, d_xq, d_b, E, (int) I, n, st);
    }, 0.0);
    proj(3, in.x, T, E, 0.0, A.iw, [&] {
        sk::dsa_gemv_f32(w.idx_proj, d_a, d_b, s.n_embd, s.idx_n_head, T, 1.0f / sqrtf((float) (I * s.idx_n_head)), st);
    }, kF32Tol);
    proj(4, A.qr, T, s.q_lora, q8_0_max_abs(hw.idx_q_b), A.iq, [&] {
        sk::native_quantize_q8_1(d_a, d_xq, (int) s.q_lora, n, st);
        sk::native_mmvq(8, w.idx_q_b, d_xq, d_b, (int) s.q_lora, (int) (s.idx_n_head * I), n, st);
    }, 0.0);
    for (int i = 0; i < kProjN; ++i) {
        agg.worst_proj[i] = std::max(agg.worst_proj[i], perr[i]);
        if (pfl[i] == 0) agg.worst_noflip[i] = std::max(agg.worst_noflip[i], perr[i]);
        agg.worst_ratio[i] = std::max(agg.worst_ratio[i], perr[i] / ptol[i]);
        agg.proj_flips[i] += pfl[i];
        agg.proj_off_rule[i] += poff[i];
        agg.proj_bad[i] += pbad[i];
        if (!(perr[i] <= ptol[i]) || pbad[i]) ok = false;
    }

    // 4. end to end: the GPU's own pooled rows again (part 2 replaced them with the oracle's), then dsa_decode
    sk::dsa_pool_range(dc.c, s, w.idx_ape, 0, p0 / P, st);
    ck(cudaMemcpy(d_x, in.x.data(), in.x.size() * 4, cudaMemcpyHostToDevice), "x");
    sk::dsa_decode(w, s, dc.c, d_x, d_steps, T, sc, d_out, st);
    ck(cudaStreamSynchronize(st), "e2e");
    const std::vector<int32_t> eids = download(sc.ids, (size_t) (T * sc.cap));
    double ov_e2e = 1.0;
    int e2e_exact = 0;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t wdt = sk::dsa_sel_width(p0 + t, s);
        const std::vector<int32_t> g(eids.begin() + t * sc.cap, eids.begin() + t * sc.cap + wdt);
        ov_e2e = std::min(ov_e2e, overlap(g, C.sel[(size_t) t]));
        e2e_exact += g == C.sel[(size_t) t];
    }
    agg.min_overlap_e2e = std::min(agg.min_overlap_e2e, ov_e2e);
    agg.e2e_tokens += T;
    agg.e2e_exact += e2e_exact;
    if (N <= sk::dsa_sel_cap(s) && e2e_exact != T) ok = false;   // dense: the identity

    std::printf("seed %llu N %5lld T %lld | scores %.1e | sel exact %d/%lld near-tie %d overlap %.4f | proj",
                (unsigned long long) seed, (long long) N, (long long) T, score_rel, exact, (long long) T, near, ov_iso);
    for (int i = 0; i < kProjN; ++i) {
        std::printf(" %s %.1e%s", kProj[i], perr[i], pfl[i] ? "*" : "");
        if (perr[i] > ptol[i]) std::printf("(>%.1e)", ptol[i]);
        if (pbad[i]) std::printf("(%lld codes off the quantizer's rule)", (long long) pbad[i]);
    }
    std::printf(" | e2e sel exact %d/%lld overlap %.4f %s\n", e2e_exact, (long long) T, ov_e2e, ok ? "ok" : "FAIL");
    std::fflush(stdout);
    agg.cases++;
    if (!ok) agg.failed++;
}

// The selection with TIES, which random scores never produce: 3,000 visible pools whose scores take 7 values (and
// -0 / +0), T = 8 tokens at different positions; the GPU's ids against a host reference (a stable sort by score,
// descending, ties to the lower pool; the top 512 pools ascending, then the tail). llama.cpp's ggml_top_k leaves
// the order of equal scores unspecified, so this pins the engine's own rule, not llama.cpp's.
bool select_ties(const DsaShapes& s, const sk::DsaScratch& sc, int32_t* d_steps, cudaStream_t st) {
    const int64_t T = 8, P = s.kpool, top = sk::dsa_top_pools(s);
    std::vector<int32_t> steps((size_t) T * sk::kStepCount);
    std::vector<float> scores((size_t) (T * sc.max_pools), 0.0f);
    std::mt19937_64 rng(4242);
    std::uniform_int_distribution<int> val(0, 6);
    std::vector<int64_t> pos(T);
    for (int64_t t = 0; t < T; ++t) {
        pos[(size_t) t] = 12000 + 333 * t + t % 4;                       // tails of 0..3 cells
        sk::dsa_step_fill(steps.data() + t * sk::kStepCount, pos[(size_t) t], 1, s);
        const int64_t nv = (pos[(size_t) t] + 1) / P;
        for (int64_t j = 0; j < nv; ++j) {
            const int v = val(rng);
            scores[(size_t) (t * sc.max_pools + j)] = v == 0 ? (j % 2 ? -0.0f : 0.0f) : (float) (v - 3) * 0.25f;
        }
    }
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
    ck(cudaMemcpy(sc.scores, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice), "scores");
    sk::dsa_select(sc.scores, s, d_steps, T, sc.max_pools, sc.cap, sc.ids, st);
    ck(cudaStreamSynchronize(st), "select");
    const std::vector<int32_t> ids = download(sc.ids, (size_t) (T * sc.cap));
    bool ok = true;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t p = pos[(size_t) t], nv = (p + 1) / P, n_tail = (p + 1) % P;
        std::vector<int32_t> order((size_t) nv);
        for (int64_t j = 0; j < nv; ++j) order[(size_t) j] = (int32_t) j;
        const float* row = scores.data() + t * sc.max_pools;
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return row[a] + 0.0f > row[b] + 0.0f; });
        order.resize((size_t) std::min(nv, top));
        std::sort(order.begin(), order.end());
        std::vector<int32_t> want;
        for (int32_t j : order) for (int64_t m = 0; m < P; ++m) want.push_back((int32_t) (j * P + m));
        for (int64_t m = n_tail - 1; m >= 0; --m) want.push_back((int32_t) (p - m));
        ok = ok && std::equal(want.begin(), want.end(), ids.begin() + t * sc.cap) &&
             (int64_t) want.size() == sk::dsa_sel_width(p, s);
    }
    std::printf("select with ties (7 score values, -0/+0, 8 tokens, 3,000+ pools): %s\n", ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    int seeds = 3, threads = 8;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--seeds" && i + 1 < argc) seeds = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--quick") quick = true;
        else { std::fprintf(stderr, "usage: %s [--seeds K] [--threads K] [--quick]\n", argv[0]); return 2; }
    }
    const DsaShapes s = sk::dsa_glm5_shapes();
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const std::vector<int64_t> Ns = quick ? std::vector<int64_t>{64, 2052} : std::vector<int64_t>{1, 64, 2051, 2052, 4096, 16384};
    const int64_t max_pools = sk::dsa_max_pools(16384, s);
    void* sbase = nullptr;
    ck(cudaMalloc(&sbase, sk::dsa_scratch_bytes(s, sk::DSA_MAX_T, max_pools)), "scratch");
    sk::DsaScratch sc;
    sk::dsa_scratch_carve(sbase, s, sk::DSA_MAX_T, max_pools, sc);
    int32_t* d_steps = nullptr;
    float *d_x = nullptr, *d_out = nullptr, *d_a = nullptr, *d_b = nullptr;
    void* d_xq = nullptr;
    const size_t big = (size_t) sk::DSA_MAX_T * s.n_embd;
    ck(cudaMalloc((void**) &d_steps, sk::DSA_MAX_T * sk::kStepCount * 4), "steps");
    ck(cudaMalloc((void**) &d_x, big * 4), "x");
    ck(cudaMalloc((void**) &d_out, big * 4), "out");
    ck(cudaMalloc((void**) &d_a, big * 4), "a");
    ck(cudaMalloc((void**) &d_b, big * 4), "b");
    ck(cudaMalloc(&d_xq, big / 32 * 36 + 256), "xq");
    Agg agg;
    const bool ties_ok = select_ties(s, sc, d_steps, st);
    const auto t0 = std::chrono::steady_clock::now();
    for (int seed = 1; seed <= seeds; ++seed) {
        const HostWeights hw = make_weights(s, (uint64_t) seed);
        const DeviceWeights dw(hw);
        for (int64_t N : Ns)
            for (int64_t T : {1, 4, 8}) {
                if (T > N) continue;
                run_case(s, hw, dw, N, T, (uint64_t) seed, sc, d_steps, d_x, d_out, d_a, d_b, d_xq, st, threads, agg);
            }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nsummary: %d cases, %d failed (%.0f s)\n", agg.cases, agg.failed, secs);
    std::printf("  key|gate rows: %lld of %lld f16 values differ; pooled rows: %lld of %lld differ (max |diff| %.2e of max "
                "|pooled|), %lld beyond one ulp or 1e-6\n", (long long) agg.kg_diff, (long long) agg.kg_total,
                (long long) agg.pooled_ulp, (long long) agg.pooled_total, agg.pooled_max_abs, (long long) agg.pooled_big);
    std::printf("  scores: worst rel. error %.2e (tolerance %.0e)\n", agg.worst_score, kScoreTol);
    std::printf("  isolated selection: %lld of %lld tokens exact, %lld near-ties, min overlap %.4f\n",
                (long long) agg.exact_tokens, (long long) agg.tokens, (long long) agg.near_tie_tokens,
                agg.min_overlap_iso);
    std::printf("  projections: worst rel. error / worst without a flipped code / error-to-tolerance ratio / codes flipped / "
                "GPU codes one step off the host's roundf(x / d) at a near-tie (CUDA's approximate division; 0 on HIP) / "
                "off the rule otherwise (must be 0):\n");
    for (int i = 0; i < kProjN; ++i)
        std::printf("    %-8s %.2e  %.2e  %.3f  %lld  %lld  %lld\n", kProj[i], agg.worst_proj[i], agg.worst_noflip[i],
                    agg.worst_ratio[i], (long long) agg.proj_flips[i], (long long) agg.proj_off_rule[i],
                    (long long) agg.proj_bad[i]);
    std::printf("  end to end selection: %lld of %lld tokens exact, min overlap %.4f\n", (long long) agg.e2e_exact,
                (long long) agg.e2e_tokens, agg.min_overlap_e2e);
    for (void* p : {(void*) sbase, (void*) d_steps, (void*) d_x, (void*) d_out, (void*) d_a, (void*) d_b, d_xq})
        (void) cudaFree(p);
    (void) cudaStreamDestroy(st);
    return agg.failed == 0 && ties_ok ? 0 : 1;
}
