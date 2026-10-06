// tests/dsa/dsa_mla_parity.cpp - the DSA layer's MLA decode step against ggml-cpu at llama.cpp f1cee99.
//
// For each seed, latent format (int8, f16), cache size N and step width T (positions N-T .. N-1; the cells before
// them are a random fixture written through the engine's own append and pooling, and through the oracle's), the
// oracle runs build_dsa_layer's ops (dsa_oracle.hpp), and:
//   1. STAGES on the oracle's own inputs, so each kernel is measured alone: wq_a, wkv_a (native Q8_0 MMVQ), the two
//      RMSNorms, wq_b, the k_b absorption (dsa_grouped_q8_0), the attention (dsa_attend over the oracle's latent
//      cache and selection; the GPU cache is first checked to hold the oracle's int8 codes and scales, or f16
//      values, bitwise), v_b and attn_output;
//   2. THE CHAINED STEP: dsa_decode from x, attending over the oracle's selection (ids_override);
//   3. END TO END: dsa_decode with its own selection: the selection against the oracle's, and `out` against the
//      oracle on the oracle's selection and (when they differ) on the GPU's.
// Errors are max |gpu - ref| / max |ref|. Exit 0 iff every case is within the tolerances below.
//
//   --seeds K   seeds 1..K (default 3)     --quick   N in {64, 2052} only     --bench   kernel times at N = 16384
//   --threads K oracle threads (default 8)
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
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace dsa_test;
namespace sk = strata::kernels;

namespace {

// Tolerances, relative to max |ref| of the output.
//
// STAGES (part 1) take the oracle's own inputs, so each kernel is measured alone:
//   - f32 arithmetic in another order: 1e-5 (the measured stage errors without a flip are 1e-7 .. 1e-6);
//   - a quantized product also quantizes its activation: the GPU to Q8_1 (code = roundf(x / d), Strata's native
//     quantizer, the same rule as llama.cpp's CUDA backend), ggml-cpu on x86 to Q8_0 (code = rint(x * (127 / amax))).
//     The codes agree except near a rounding tie; each disagreement moves one code by 1, i.e. the column's output by
//     at most d * max|w|. Which way a near-tie goes is the GPU's own arithmetic (CUDA's quantizer divides with
//     div.approx under --use_fast_math, HIP in IEEE), so `q8_1_flip_bound` reads the GPU's Q8_1 blocks back after
//     the product and compares them with ggml-cpu's codes of the same values: the tolerance is
//     1e-5 + 1.5 * (sum of |d * code difference| over the column's flipped codes) * max|w| / max|ref|, exactly 1e-5
//     when no code flips, and a bound derived from the data when one does. The read-back codes are themselves held
//     to the rule: each must be roundf(x / d) of the host, or one step off it at a near-tie (counted, "off rule");
//     anything else fails the case, so a broken quantizer cannot widen its own tolerance.
// THE CHAINED STEP (part 2, dsa_decode on the oracle's selection) is a wiring check: every stage is held to its own
// tolerance above, and a chain without a flipped code agrees to ~1e-6 (counted as "flip-free"); a flip early in
// the chain (in x, before wq_a) is carried through five quantized products and the softmax, and with these random
// weights moves `out` by up to ~2e-2 of its max. A wiring error (a wrong head, scale, mask or layout) is O(1).
// So 5e-2. Part 3 (the GPU's own selection) is reported, and its MLA error on that selection is held to the same.
constexpr double kF32Tol = 1e-5, kNormTol = 1e-6, kChainTol = 5e-2, kFlipFree = 1e-5;

struct StageErr {
    const char* name;
    double rel = 0.0, tol = 0.0;
    int64_t flips = 0, off_rule = 0, bad = 0;
};
constexpr int kStages = 9;
constexpr const char* kStageNames[kStages] = {"wq_a", "wkv_a", "q_norm", "kv_norm", "wq_b", "k_b", "attend", "v_b", "wo"};
constexpr int kChain = 7;
constexpr const char* kChainNames[kChain] = {"qr", "q", "q_abs", "latent", "o_lat", "o", "out"};

struct Agg {
    double worst[kStages] = {}, worst_ratio[kStages] = {}, worst_noflip[kStages] = {};
    int64_t flips[kStages] = {}, off_rule[kStages] = {}, bad[kStages] = {};
    double worst_chain[kChain] = {};
    double worst_e2e_out = 0.0, worst_e2e_mla = 0.0, min_overlap = 1.0;
    int64_t code_mismatch = 0, code_total = 0;
    int cases = 0, failed = 0, sel_diff = 0, flip_free = 0;
};

const char* fmt_name(int fmt) { return fmt == sk::kDsaLatInt8 ? "int8" : "f16"; }

struct Bufs {
    float *a = nullptr, *b = nullptr, *c = nullptr;   // device scratch for stage inputs/outputs
    void* xq = nullptr;
};

// one quantized product's stage check: x [cols][n_in] (host, the oracle's values) through `run` on the device,
// which quantizes it into d_xq (read back for the flip bound) and multiplies into d_y
StageErr stage_q(const char* name, const std::vector<float>& x, int64_t cols, int64_t n_in, double max_w,
                 const std::vector<float>& ref, float* d_x, const void* d_xq, float* d_y, cudaStream_t st,
                 const std::function<void()>& run) {
    StageErr e{name};
    ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "stage in");
    run();
    ck(cudaStreamSynchronize(st), "stage");
    const std::vector<float> got = download(d_y, ref.size());
    const Err r = compare(got.data(), ref.data(), (int64_t) ref.size());
    const std::vector<uint8_t> xq = download(static_cast<const uint8_t*>(d_xq), (size_t) (cols * (n_in / 32) * 36));
    const FlipBound fb = q8_1_flip_bound(x.data(), xq.data(), cols, n_in);
    e.flips = fb.flips;
    e.off_rule = fb.off_rule;
    e.bad = fb.bad;
    e.rel = r.rel;
    e.tol = kF32Tol + 1.5 * fb.steps * max_w / (r.ref_max > 0 ? r.ref_max : 1.0);
    return e;
}
StageErr stage_f(const char* name, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
    StageErr e{name};
    e.rel = compare(got.data(), ref.data(), (int64_t) ref.size()).rel;
    e.tol = tol;
    return e;
}

void run_case(const DsaShapes& s, const HostWeights& hw, const DeviceWeights& dw, int fmt, int64_t N, int64_t T,
              uint64_t seed, const sk::DsaScratch& sc, int32_t* d_steps, int32_t* d_ids, float* d_x, float* d_out,
              const Bufs& bf, cudaStream_t st, int threads, Agg& agg) {
    const int64_t p0 = N - T;
    const CaseInputs in = make_case(s, hw, p0, T, seed * 1000003ull + (uint64_t) N * 31 + (uint64_t) T);
    DeviceCache dc(s, fmt, N, st);
    HostCache hc;
    cache_init(hc, s, fmt, N);
    fill_old(s, hw, dw.w, in, p0, dc.c, hc, st, threads);

    // the oracle's step
    StepOut A, C, D;
    oracle_project(hw, s, in.x, T, A, threads);
    cache_append(hc, s, p0, T, A.latent.data(), A.ik.data(), A.ig.data());
    oracle_pool(hc, s, hw.idx_ape, p0 / s.kpool, (p0 + T) / s.kpool, threads);
    oracle_select(hc, s, A, p0, T, C, threads);
    oracle_attend(hw, hc, s, A, p0, T, C.slots, D, threads);

    std::vector<int32_t> steps((size_t) T * sk::kStepCount);
    sk::dsa_step_fill(steps.data(), p0, T, s);
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
    const std::vector<int32_t> oids = id_rows(C.sel, sc.cap);
    ck(cudaMemcpy(d_ids, oids.data(), oids.size() * 4, cudaMemcpyHostToDevice), "ids");
    const int E = (int) s.n_embd, n = (int) T, H = (int) s.n_head;
    const auto& w = dw.w;

    // ---- part 1: every stage on the oracle's inputs
    StageErr se[kStages];
    se[0] = stage_q("wq_a", in.x, T, E, q8_0_max_abs(hw.wq_a), A.qa, bf.a, bf.xq, bf.b, st, [&] {
        sk::native_quantize_q8_1(bf.a, bf.xq, E, n, st);
        sk::native_mmvq(8, w.wq_a, bf.xq, bf.b, E, (int) s.q_lora, n, st);
    });
    se[1] = stage_q("wkv_a", in.x, T, E, q8_0_max_abs(hw.wkv_a), A.kv, bf.a, bf.xq, bf.b, st, [&] {
        sk::native_quantize_q8_1(bf.a, bf.xq, E, n, st);
        sk::native_mmvq(8, w.wkv_a, bf.xq, bf.b, E, (int) s.kv_lora, n, st);
    });
    {
        ck(cudaMemcpy(bf.a, A.qa.data(), A.qa.size() * 4, cudaMemcpyHostToDevice), "qa");
        sk::dsa_rms_norm(bf.a, w.q_a_norm, s.rms_eps, s.q_lora, T, bf.b, st);
        se[2] = stage_f("q_norm", download(bf.b, A.qr.size()), A.qr, kNormTol);
        ck(cudaMemcpy(bf.a, A.kv.data(), A.kv.size() * 4, cudaMemcpyHostToDevice), "kv");
        sk::dsa_rms_norm(bf.a, w.kv_a_norm, s.rms_eps, s.kv_lora, T, bf.b, st);
        se[3] = stage_f("kv_norm", download(bf.b, A.latent.size()), A.latent, kNormTol);
    }
    se[4] = stage_q("wq_b", A.qr, T, s.q_lora, q8_0_max_abs(hw.wq_b), A.q, bf.a, bf.xq, bf.b, st, [&] {
        sk::native_quantize_q8_1(bf.a, bf.xq, (int) s.q_lora, n, st);
        sk::native_mmvq(8, w.wq_b, bf.xq, bf.b, (int) s.q_lora, (int) (s.n_head * s.head_k), n, st);
    });
    se[5] = stage_q("k_b", A.q, T * s.n_head, s.head_k, q8_0_max_abs(hw.wk_b), A.q_abs, bf.a, bf.xq, bf.b, st, [&] {
        sk::native_quantize_q8_1(bf.a, bf.xq, (int) (T * s.n_head * s.head_k), 1, st);
        sk::dsa_grouped_q8_0(w.wk_b, bf.xq, bf.b, s.head_k, s.kv_lora, s.n_head, T, st);
    });
    // attention on the oracle's latent and q_abs: the window's cells get the oracle's latent, so the GPU cache
    // holds exactly what the oracle's does (checked: int8 codes and scales of every cell, bitwise)
    int64_t mism = 0, total = 0;
    {
        float* d_lat = upload(A.latent);
        float* d_ik = upload(A.ik);
        float* d_ig = upload(A.ig);
        sk::dsa_kv_append(dc.c, s, d_steps, T, d_lat, d_ik, d_ig, st);
        ck(cudaStreamSynchronize(st), "append");
        (void) cudaFree(d_lat);
        (void) cudaFree(d_ik);
        (void) cudaFree(d_ig);
        if (fmt == sk::kDsaLatInt8) {
            const std::vector<int8_t> gc = download(dc.c.lat_q, (size_t) (N * s.kv_lora));
            const std::vector<uint16_t> gs = download(dc.c.lat_scale, (size_t) (N * s.kv_lora / sk::DSA_LAT_GROUP));
            for (size_t i = 0; i < gc.size(); ++i) mism += gc[i] != hc.codes[i];
            for (size_t i = 0; i < gs.size(); ++i) mism += gs[i] != hc.scales[i];
            total = (int64_t) (gc.size() + gs.size());
        } else {
            const std::vector<uint16_t> gh = download(dc.c.lat_f16, (size_t) (N * s.kv_lora));
            for (size_t i = 0; i < gh.size(); ++i) mism += sk::f32_from_f16(gh[i]) != hc.latent[i];
            total = (int64_t) gh.size();
        }
        ck(cudaMemcpy(bf.a, A.q_abs.data(), A.q_abs.size() * 4, cudaMemcpyHostToDevice), "q_abs");
        sk::dsa_attend(dc.c, s, bf.a, d_ids, d_steps, T, sc.cap, 1.0f / sqrtf((float) s.head_k), sc.attn, bf.b, st);
        se[6] = stage_f("attend", download(bf.b, D.o_lat.size()), D.o_lat, kF32Tol);
    }
    se[7] = stage_q("v_b", D.o_lat, T * s.n_head, s.kv_lora, q8_0_max_abs(hw.wv_b), D.o, bf.a, bf.xq, bf.b, st, [&] {
        sk::native_quantize_q8_1(bf.a, bf.xq, (int) (T * s.n_head * s.kv_lora), 1, st);
        sk::dsa_grouped_q8_0(w.wv_b, bf.xq, bf.b, s.kv_lora, s.head_v, s.n_head, T, st);
    });
    se[8] = stage_q("wo", D.o, T, s.n_head * s.head_v, q8_0_max_abs(hw.wo), D.out, bf.a, bf.xq, bf.b, st, [&] {
        sk::native_quantize_q8_1(bf.a, bf.xq, H * (int) s.head_v, n, st);
        sk::native_mmvq(8, w.wo, bf.xq, bf.b, H * (int) s.head_v, E, n, st);
    });

    // ---- part 2: the chained step on the oracle's selection
    ck(cudaMemcpy(d_x, in.x.data(), in.x.size() * 4, cudaMemcpyHostToDevice), "x");
    sk::dsa_decode(w, s, dc.c, d_x, d_steps, T, sc, d_out, st, d_ids);
    ck(cudaStreamSynchronize(st), "decode");
    const size_t nq = (size_t) (T * s.n_head);
    double chain[kChain];
    {
        const std::vector<float> g_qr = download(sc.qr, (size_t) (T * s.q_lora));
        const std::vector<float> g_q = download(sc.q, nq * s.head_k);
        const std::vector<float> g_qabs = download(sc.q_abs, nq * s.kv_lora);
        const std::vector<float> g_lat = download(sc.latent, (size_t) (T * s.kv_lora));
        const std::vector<float> g_olat = download(sc.o_lat, nq * s.kv_lora);
        const std::vector<float> g_o = download(sc.o, nq * s.head_v);
        const std::vector<float> g_out = download(d_out, (size_t) (T * s.n_embd));
        chain[0] = compare(g_qr.data(), A.qr.data(), (int64_t) g_qr.size()).rel;
        chain[1] = compare(g_q.data(), A.q.data(), (int64_t) g_q.size()).rel;
        chain[2] = compare(g_qabs.data(), A.q_abs.data(), (int64_t) g_qabs.size()).rel;
        chain[3] = compare(g_lat.data(), A.latent.data(), (int64_t) g_lat.size()).rel;
        chain[4] = compare(g_olat.data(), D.o_lat.data(), (int64_t) g_olat.size()).rel;
        chain[5] = compare(g_o.data(), D.o.data(), (int64_t) g_o.size()).rel;
        chain[6] = compare(g_out.data(), D.out.data(), (int64_t) g_out.size()).rel;
    }

    // ---- part 3: end to end, the GPU's own selection
    sk::dsa_decode(w, s, dc.c, d_x, d_steps, T, sc, d_out, st, nullptr);
    ck(cudaStreamSynchronize(st), "decode e2e");
    const std::vector<int32_t> gids = download(sc.ids, (size_t) (T * sc.cap));
    const std::vector<float> g_out2 = download(d_out, (size_t) (T * s.n_embd));
    std::vector<std::vector<int32_t>> gsel((size_t) T);
    double ov = 1.0;
    bool same_sel = true;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t wdt = sk::dsa_sel_width(p0 + t, s);
        gsel[(size_t) t].assign(gids.begin() + t * sc.cap, gids.begin() + t * sc.cap + wdt);
        ov = std::min(ov, overlap(gsel[(size_t) t], C.sel[(size_t) t]));
        same_sel = same_sel && gsel[(size_t) t] == C.sel[(size_t) t];
    }
    const Err e2e = compare(g_out2.data(), D.out.data(), (int64_t) g_out2.size());
    Err e2e_mla = e2e;
    if (!same_sel) {   // the oracle on the GPU's selection: the MLA error alone
        StepOut D2;
        oracle_attend(hw, hc, s, A, p0, T, slots_of(gsel, (int64_t) C.slots[0].size()), D2, threads);
        e2e_mla = compare(g_out2.data(), D2.out.data(), (int64_t) g_out2.size());
    }

    bool ok = mism == 0;
    std::printf("seed %llu %-4s N %5lld T %lld | stages", (unsigned long long) seed, fmt_name(fmt), (long long) N,
                (long long) T);
    for (int i = 0; i < kStages; ++i) {
        const bool pass = se[i].rel <= se[i].tol && se[i].bad == 0;
        std::printf(" %s %.1e%s", se[i].name, se[i].rel, se[i].flips ? "*" : "");
        if (se[i].rel > se[i].tol) std::printf("(>%.1e)", se[i].tol);
        if (se[i].bad) std::printf("(%lld codes off the quantizer's rule)", (long long) se[i].bad);
        if (!pass) ok = false;
        agg.worst[i] = std::max(agg.worst[i], se[i].rel);
        if (se[i].flips == 0) agg.worst_noflip[i] = std::max(agg.worst_noflip[i], se[i].rel);
        agg.worst_ratio[i] = std::max(agg.worst_ratio[i], se[i].rel / se[i].tol);
        agg.flips[i] += se[i].flips;
        agg.off_rule[i] += se[i].off_rule;
        agg.bad[i] += se[i].bad;
    }
    std::printf(" | chain");
    double chain_max = 0.0;
    for (int i = 0; i < kChain; ++i) {
        std::printf(" %s %.1e", kChainNames[i], chain[i]);
        agg.worst_chain[i] = std::max(agg.worst_chain[i], chain[i]);
        chain_max = std::max(chain_max, chain[i]);
        if (!(chain[i] <= kChainTol)) ok = false;
    }
    agg.flip_free += chain_max <= kFlipFree;
    std::printf(" | own sel %s overlap %.4f out %.1e mla %.1e", same_sel ? "same" : "DIFF", ov, e2e.rel, e2e_mla.rel);
    if (mism) std::printf(" | cache MISMATCH %lld/%lld", (long long) mism, (long long) total);
    if (!(e2e_mla.rel <= kChainTol)) ok = false;
    if (N <= sk::dsa_sel_cap(s) && !same_sel) ok = false;   // the dense regime: the selection is the identity
    std::printf(" %s\n", ok ? "ok" : "FAIL");
    std::fflush(stdout);
    agg.worst_e2e_out = std::max(agg.worst_e2e_out, e2e.rel);
    agg.worst_e2e_mla = std::max(agg.worst_e2e_mla, e2e_mla.rel);
    agg.min_overlap = std::min(agg.min_overlap, ov);
    agg.code_mismatch += mism;
    agg.code_total += total;
    agg.sel_diff += !same_sel;
    agg.cases++;
    if (!ok) agg.failed++;
}

// Per-kernel times for one token at N = 16384 (int8), median of 100 event-timed launches each.
void bench(const DsaShapes& s, const HostWeights& hw, const DeviceWeights& dw, const sk::DsaScratch& sc,
           int32_t* d_steps, float* d_x, float* d_out, cudaStream_t st, int threads) {
    const int64_t N = 16384, T = 1, p0 = N - T;
    const CaseInputs in = make_case(s, hw, p0, T, 777);
    DeviceCache dc(s, sk::kDsaLatInt8, N, st);
    HostCache hc;
    cache_init(hc, s, sk::kDsaLatInt8, N);
    fill_old(s, hw, dw.w, in, p0, dc.c, hc, st, threads);
    std::vector<int32_t> steps((size_t) T * sk::kStepCount);
    sk::dsa_step_fill(steps.data(), p0, T, s);
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
    ck(cudaMemcpy(d_x, in.x.data(), in.x.size() * 4, cudaMemcpyHostToDevice), "x");
    sk::dsa_decode(dw.w, s, dc.c, d_x, d_steps, T, sc, d_out, st);   // warm, and fills the scratch
    ck(cudaStreamSynchronize(st), "warm");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto time = [&](const char* name, auto&& fn) {
        std::vector<float> ms;
        for (int i = 0; i < 110; ++i) {
            ck(cudaEventRecord(e0, st), "rec");
            fn();
            ck(cudaEventRecord(e1, st), "rec");
            ck(cudaEventSynchronize(e1), "sync");
            float m = 0.0f;
            ck(cudaEventElapsedTime(&m, e0, e1), "elapsed");
            if (i >= 10) ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("  %-34s %8.1f us\n", name, 1000.0 * ms[ms.size() / 2]);
    };
    const auto& w = dw.w;
    const int E = (int) s.n_embd;
    std::printf("kernel times, one token, N = %lld cells, int8 latent (median of 100):\n", (long long) N);
    time("quantize x (q8_1)", [&] { sk::native_quantize_q8_1(d_x, sc.xq, E, 1, st); });
    time("wq_a 4096->1536 (Q8_0 mmvq)", [&] { sk::native_mmvq(8, w.wq_a, sc.xq, sc.qa, E, (int) s.q_lora, 1, st); });
    time("wkv_a 4096->512", [&] { sk::native_mmvq(8, w.wkv_a, sc.xq, sc.kv, E, (int) s.kv_lora, 1, st); });
    time("idx_k + idx_gate 4096->128 x2", [&] {
        sk::native_mmvq(8, w.idx_k, sc.xq, sc.ik_raw, E, (int) s.idx_dim, 1, st);
        sk::native_mmvq(8, w.idx_gate, sc.xq, sc.ig, E, (int) s.idx_dim, 1, st);
    });
    time("idx_proj 4096->32 (f32)", [&] { sk::dsa_gemv_f32(w.idx_proj, d_x, sc.iw, s.n_embd, s.idx_n_head, 1, 0.015625f, st); });
    time("norms (rms x2, layer)", [&] {
        sk::dsa_rms_norm(sc.qa, w.q_a_norm, s.rms_eps, s.q_lora, 1, sc.qr, st);
        sk::dsa_rms_norm(sc.kv, w.kv_a_norm, s.rms_eps, s.kv_lora, 1, sc.latent, st);
        sk::dsa_layer_norm(sc.ik_raw, w.idx_k_norm, w.idx_k_norm_b, s.norm_eps, s.idx_dim, 1, sc.ik, st);
    });
    time("quantize qr + wq_b 1536->16384", [&] {
        sk::native_quantize_q8_1(sc.qr, sc.xq, (int) s.q_lora, 1, st);
        sk::native_mmvq(8, w.wq_b, sc.xq, sc.q, (int) s.q_lora, (int) (s.n_head * s.head_k), 1, st);
    });
    time("idx_q_b 1536->4096", [&] {
        sk::native_mmvq(8, w.idx_q_b, sc.xq, sc.iq, (int) s.q_lora, (int) (s.idx_n_head * s.idx_dim), 1, st);
    });
    time("k_b absorb (quantize + grouped)", [&] {
        sk::native_quantize_q8_1(sc.q, sc.xq, (int) (s.n_head * s.head_k), 1, st);
        sk::dsa_grouped_q8_0(w.wk_b, sc.xq, sc.q_abs, s.head_k, s.kv_lora, s.n_head, 1, st);
    });
    time("kv append (int8 + key|gate)", [&] { sk::dsa_kv_append(dc.c, s, d_steps, 1, sc.latent, sc.ik, sc.ig, st); });
    time("pool step", [&] { sk::dsa_pool_step(dc.c, s, w.idx_ape, d_steps, 1, st); });
    time("index scores (4096 pools)", [&] { sk::dsa_index_scores(dc.c, s, sc.iq, sc.iw, d_steps, 1, sc.max_pools, sc.scores, st); });
    time("select (top 512 of 4096 pools)", [&] {
        sk::dsa_select(sc.scores, s, d_steps, 1, sc.max_pools, sc.cap, sc.ids, st);
    });
    time("attend (2,048 cells, 64 heads)", [&] {
        sk::dsa_attend(dc.c, s, sc.q_abs, sc.ids, d_steps, 1, sc.cap, 0.0625f, sc.attn, sc.o_lat, st);
    });
    time("v_b (quantize + grouped)", [&] {
        sk::native_quantize_q8_1(sc.o_lat, sc.xq, (int) (s.n_head * s.kv_lora), 1, st);
        sk::dsa_grouped_q8_0(w.wv_b, sc.xq, sc.o, s.kv_lora, s.head_v, s.n_head, 1, st);
    });
    time("attn_output (quantize + 16384->4096)", [&] {
        sk::native_quantize_q8_1(sc.o, sc.xq, (int) (s.n_head * s.head_v), 1, st);
        sk::native_mmvq(8, w.wo, sc.xq, d_out, (int) (s.n_head * s.head_v), E, 1, st);
    });
    time("dsa_decode, whole layer", [&] { sk::dsa_decode(w, s, dc.c, d_x, d_steps, 1, sc, d_out, st); });
    (void) cudaEventDestroy(e0);
    (void) cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    int seeds = 3, threads = 8;
    bool quick = false, do_bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--seeds" && i + 1 < argc) seeds = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--quick") quick = true;
        else if (a == "--bench") do_bench = true;
        else { std::fprintf(stderr, "usage: %s [--seeds K] [--threads K] [--quick] [--bench]\n", argv[0]); return 2; }
    }
    const DsaShapes s = sk::dsa_glm5_shapes();
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const std::vector<int64_t> Ns = quick ? std::vector<int64_t>{64, 2052} : std::vector<int64_t>{1, 64, 2051, 2052, 4096, 16384};
    const std::vector<int64_t> Ts = {1, 4, 8};
    const int64_t max_pools = sk::dsa_max_pools(16384, s);
    void* sbase = nullptr;
    ck(cudaMalloc(&sbase, sk::dsa_scratch_bytes(s, sk::DSA_MAX_T, max_pools)), "scratch");
    sk::DsaScratch sc;
    sk::dsa_scratch_carve(sbase, s, sk::DSA_MAX_T, max_pools, sc);
    int32_t* d_steps = nullptr;
    int32_t* d_ids = nullptr;
    float* d_x = nullptr;
    float* d_out = nullptr;
    ck(cudaMalloc((void**) &d_steps, sk::DSA_MAX_T * sk::kStepCount * 4), "steps");
    ck(cudaMalloc((void**) &d_ids, (size_t) sk::DSA_MAX_T * sc.cap * 4), "ids");
    ck(cudaMalloc((void**) &d_x, (size_t) sk::DSA_MAX_T * s.n_embd * 4), "x");
    ck(cudaMalloc((void**) &d_out, (size_t) sk::DSA_MAX_T * s.n_embd * 4), "out");
    Bufs bf;
    const size_t big = (size_t) sk::DSA_MAX_T * s.n_head * s.kv_lora;
    ck(cudaMalloc((void**) &bf.a, big * 4), "bufs");
    ck(cudaMalloc((void**) &bf.b, big * 4), "bufs");
    ck(cudaMalloc(&bf.xq, big / 32 * 36 + 256), "bufs");
    Agg agg;
    const auto t0 = std::chrono::steady_clock::now();
    for (int seed = 1; seed <= seeds; ++seed) {
        const HostWeights hw = make_weights(s, (uint64_t) seed);
        const DeviceWeights dw(hw);
        for (int fmt : {(int) sk::kDsaLatInt8, (int) sk::kDsaLatF16})
            for (int64_t N : Ns)
                for (int64_t T : Ts) {
                    if (T > N) continue;
                    run_case(s, hw, dw, fmt, N, T, (uint64_t) seed, sc, d_steps, d_ids, d_x, d_out, bf, st, threads, agg);
                }
        if (do_bench && seed == 1) bench(s, hw, dw, sc, d_steps, d_x, d_out, st, threads);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nsummary: %d cases, %d failed (%.0f s)\n", agg.cases, agg.failed, secs);
    std::printf("  stages on the oracle's inputs: worst rel. error / worst without a flipped code / worst error-to-tolerance "
                "ratio / codes flipped / GPU codes one step off the host's roundf(x / d) at a near-tie (CUDA's "
                "approximate division; 0 on HIP) / off the rule otherwise (must be 0):\n");
    for (int i = 0; i < kStages; ++i)
        std::printf("    %-8s %.2e  %.2e  %.3f  %lld  %lld  %lld\n", kStageNames[i], agg.worst[i], agg.worst_noflip[i],
                    agg.worst_ratio[i], (long long) agg.flips[i], (long long) agg.off_rule[i], (long long) agg.bad[i]);
    std::printf("  chained step (tolerance %.0e; %d of %d cases flip-free, every output <= %.0e):", kChainTol,
                agg.flip_free, agg.cases, kFlipFree);
    for (int i = 0; i < kChain; ++i) std::printf(" %s %.2e", kChainNames[i], agg.worst_chain[i]);
    std::printf("\n  own selection: %d case(s) differ from the oracle's, min overlap %.4f; out vs oracle %.2e, MLA on "
                "the GPU's selection %.2e\n", agg.sel_diff, agg.min_overlap, agg.worst_e2e_out, agg.worst_e2e_mla);
    std::printf("  cache contents vs the oracle's (int8 codes + scales, f16 values): %lld of %lld differ\n",
                (long long) agg.code_mismatch, (long long) agg.code_total);
    for (void* p : {(void*) sbase, (void*) d_steps, (void*) d_ids, (void*) d_x, (void*) d_out, (void*) bf.a, (void*) bf.b, bf.xq})
        (void) cudaFree(p);
    (void) cudaStreamDestroy(st);
    return agg.failed == 0 ? 0 : 1;
}
