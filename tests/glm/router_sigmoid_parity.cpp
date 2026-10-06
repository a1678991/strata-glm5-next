// tests/glm/router_sigmoid_parity.cpp - GLM-5.3-Flash's sigmoid router with the exp_probs_b selection bias.
//
// Oracles:
//   ggml-cpu at llama.cpp f1cee99, the ops of build_moe_ffn (src/llama-graph.cpp 2069-2173) as a graph: mul_mat on
//   the F32 router, ggml_sigmoid, ggml_add(exp_probs_b), ggml_argsort_top_k, ggml_get_rows, ggml_sum_rows,
//   ggml_clamp(6.103515625e-5), ggml_div, ggml_scale(2.5);
//   the scalar transcription of the same lines, include/strata/kernels/cpu/router_sigmoid.hpp (the host router the
//   lookahead calls), itself checked against the ggml graph here.
// Checks (288 experts, top 8, 4096-wide router, 4096 tokens per seed, seeds 1..3):
//   logits (F32 GEMV) against ggml mul_mat      tol 1e-5 of max |ref| (4096-term F32 dot, different summation order)
//   GPU top-k on the ORACLE's logits             ids identical in order; weights tol 1e-6 (division and one
//                                                multiplication after an expf that may differ by an ulp)
//   host router on the oracle's logits           ids identical in order; weights tol 1e-6
//   the whole chain (GPU logits -> GPU top-k)    every id set that differs from the oracle's must be a near tie: the
//                                                oracle's k-th and (k+1)-th selection values within 1e-5 (the logits'
//                                                own error); their count is printed
//   ties (constructed): equal selection values pick the lower expert index on GPU and host alike
//   a negative control: the GPU without the bias must select differently from the oracle on most tokens.
#include "strata/kernels/router_sigmoid.hpp"
#include "strata/kernels/cpu/router_sigmoid.hpp"

#include "ggml_oracle.hpp"
#include "glm_test_util.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace glmtest;
namespace k = strata::kernels;

namespace {

constexpr int NE = 4096, E = 288, K = 8;

bool run(unsigned seed, const k::SigmoidRouterShape& sh, cudaStream_t st, bool timing) {
    const int T = 4096;
    std::printf("[router] seed %u, %d tokens\n", seed, T);
    std::mt19937 rng(seed * 15485863u);
    const auto x = normal(rng, (size_t) T * NE, 1.0f);
    const auto w = normal(rng, (size_t) E * NE, 1.5f / 64.0f);       // logits ~ N(0, 1.5^2)
    const auto bias = normal(rng, E, 0.05f);                          // exp_probs_b ~ the probability scale

    Oracle o((size_t) 512 << 20);
    ggml_context* ctx = o.ctx;
    ggml_tensor* X = o.f32(x, NE, T);
    ggml_tensor* W = o.f32(w, NE, E);
    ggml_tensor* logits = ggml_mul_mat(ctx, W, X);                                   // [E, T]
    ggml_tensor* probs = ggml_sigmoid(ctx, logits);                                  // 2069
    ggml_tensor* selp = ggml_add(ctx, probs, o.f32(bias, E));                        // 2088
    ggml_tensor* sel = ggml_argsort_top_k(ctx, selp, K);                             // 2131
    ggml_tensor* wts = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, E, T), sel); // 2145
    wts = ggml_reshape_2d(ctx, wts, K, T);
    ggml_tensor* sum = ggml_clamp(ctx, ggml_sum_rows(ctx, wts), 6.103515625e-5f, INFINITY);   // 2159, 2163
    wts = ggml_div(ctx, wts, sum);                                                   // 2166
    wts = ggml_scale(ctx, wts, sh.scale);                                            // 2172
    sel = ggml_cont(ctx, sel);
    o.compute({logits, selp, sel, wts});
    const auto r_logits = Oracle::read(logits), r_sel = Oracle::read(selp), r_w = Oracle::read(wts);
    std::vector<int32_t> r_ids((size_t) T * K);
    std::memcpy(r_ids.data(), sel->data, r_ids.size() * 4);

    Dev<float> dx(x), dw(w), db(bias), dl((size_t) T * E), dl_ref(r_logits), dwt((size_t) T * K);
    Dev<int32_t> did((size_t) T * K);
    bool ok = true;

    // logits
    k::router_f32_logits(dw.p, dx.p, dl.p, sh, T, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    const auto g_logits = dl.down();
    ok &= report("logits = W x (F32)", compare(g_logits.data(), r_logits.data(), g_logits.size()), 1e-5);

    // top-k on the oracle's logits
    k::router_sigmoid_topk(dl_ref.p, db.p, did.p, dwt.p, sh, T, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    auto g_ids = did.down();
    auto g_w = dwt.down();
    {
        size_t bad = 0;
        for (size_t i = 0; i < g_ids.size(); ++i) bad += g_ids[i] != r_ids[i];
        std::printf("  %-34s %zu of %zu ids differ  %s\n", "GPU top-k ids (oracle logits)", bad, g_ids.size(),
                    bad == 0 ? "PASS" : "FAIL");
        ok &= bad == 0;
        ok &= report("GPU weights (oracle logits)", compare(g_w.data(), r_w.data(), g_w.size()), 1e-6);
    }
    // the host router on the oracle's logits
    {
        std::vector<int32_t> h_ids((size_t) T * K);
        std::vector<float> h_w((size_t) T * K), probs;
        std::vector<int32_t> order;
        for (int t = 0; t < T; ++t)
            k::cpu::sigmoid_route(r_logits.data() + (size_t) t * E, bias.data(), E, K, sh.norm, sh.scale,
                                  h_ids.data() + (size_t) t * K, h_w.data() + (size_t) t * K, probs, order);
        size_t bad = 0;
        for (size_t i = 0; i < h_ids.size(); ++i) bad += h_ids[i] != r_ids[i];
        std::printf("  %-34s %zu of %zu ids differ  %s\n", "host router ids (oracle logits)", bad, h_ids.size(),
                    bad == 0 ? "PASS" : "FAIL");
        ok &= bad == 0;
        ok &= report("host router weights", compare(h_w.data(), r_w.data(), h_w.size()), 1e-6);
    }
    // the whole chain
    {
        k::router_sigmoid_topk(dl.p, db.p, did.p, dwt.p, sh, T, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        g_ids = did.down();
        int differ = 0, unexplained = 0;
        double worst_gap = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<int32_t> a(g_ids.begin() + (size_t) t * K, g_ids.begin() + (size_t) (t + 1) * K);
            std::vector<int32_t> b(r_ids.begin() + (size_t) t * K, r_ids.begin() + (size_t) (t + 1) * K);
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a == b) continue;
            ++differ;
            std::vector<float> v(r_sel.begin() + (size_t) t * E, r_sel.begin() + (size_t) (t + 1) * E);
            std::sort(v.begin(), v.end(), std::greater<float>());
            const double gap = v[K - 1] - v[K];
            worst_gap = std::max(worst_gap, gap);
            if (gap > 1e-5) ++unexplained;
        }
        const bool p = unexplained == 0;
        std::printf("  %-34s %d of %d tokens select another set, all near ties (largest k/k+1 gap %.2e): %s\n",
                    "chain GPU logits -> GPU top-k", differ, T, worst_gap, p ? "PASS" : "FAIL");
        ok &= p;
    }
    // negative control: no bias
    {
        k::router_sigmoid_topk(dl_ref.p, nullptr, did.p, dwt.p, sh, T, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        g_ids = did.down();
        int differ = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<int32_t> a(g_ids.begin() + (size_t) t * K, g_ids.begin() + (size_t) (t + 1) * K);
            std::vector<int32_t> b(r_ids.begin() + (size_t) t * K, r_ids.begin() + (size_t) (t + 1) * K);
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            differ += a != b;
        }
        const bool p = differ > T / 2;
        std::printf("  %-34s %d of %d tokens differ without exp_probs_b: %s\n", "control: bias dropped", differ, T,
                    p ? "detected (PASS)" : "NOT detected (FAIL)");
        ok &= p;
    }
    if (timing) {
        std::printf("  [timing] median of 100, best of 5 rounds\n");
    std::printf("    (floor: a 4-byte memset launch takes %.2f us here)\n", floor_us(st));
        std::printf("    router_f32_logits    1 token   %8.2f us\n",
                    median_us([&] { k::router_f32_logits(dw.p, dx.p, dl.p, sh, 1, st); }, st));
        std::printf("    router_f32_logits    8 tokens  %8.2f us\n",
                    median_us([&] { k::router_f32_logits(dw.p, dx.p, dl.p, sh, 8, st); }, st));
        std::printf("    router_sigmoid_topk  1 token   %8.2f us\n",
                    median_us([&] { k::router_sigmoid_topk(dl.p, db.p, did.p, dwt.p, sh, 1, st); }, st));
        std::printf("    router_sigmoid_topk  8 tokens  %8.2f us\n",
                    median_us([&] { k::router_sigmoid_topk(dl.p, db.p, did.p, dwt.p, sh, 8, st); }, st));
    }
    return ok;
}

bool ties(const k::SigmoidRouterShape& sh, cudaStream_t st) {
    std::printf("[router] constructed ties\n");
    // logits all 0 -> p = 0.5 everywhere; bias equal on blocks of experts -> many exact ties in sel
    std::vector<float> lg((size_t) 4 * E, 0.0f), bias(E);
    for (int e = 0; e < E; ++e) bias[e] = (float) ((e * 7) % 5) * 0.25f;
    for (int e = 0; e < E; ++e) lg[(size_t) 3 * E + e] = (e % 3 == 0) ? 1.0f : 0.0f;
    Dev<float> dl(lg), db(bias), dw((size_t) 4 * K);
    Dev<int32_t> di((size_t) 4 * K);
    k::router_sigmoid_topk(dl.p, db.p, di.p, dw.p, sh, 4, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    const auto g = di.down();
    const auto gw = dw.down();
    std::vector<int32_t> h((size_t) 4 * K);
    std::vector<float> hw((size_t) 4 * K), probs;
    std::vector<int32_t> order;
    for (int t = 0; t < 4; ++t)
        k::cpu::sigmoid_route(lg.data() + (size_t) t * E, bias.data(), E, K, true, sh.scale, h.data() + (size_t) t * K,
                              hw.data() + (size_t) t * K, probs, order);
    const bool same = g == h;
    std::printf("  %-34s %s (first ids %d %d %d ... )\n", "GPU ids == host ids on ties", same ? "PASS" : "FAIL", g[0],
                g[1], g[2]);
    return same && report("GPU weights vs host on ties", compare(gw.data(), hw.data(), hw.size()), 1e-6);
}

}  // namespace

int main() {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    k::SigmoidRouterShape sh;   // 4096, 288 experts, top 8, norm, scale 2.5
    bool ok = true;
    for (unsigned seed = 1; seed <= 3; ++seed) ok &= run(seed, sh, st, seed == 1);
    ok &= ties(sh, st);
    std::printf("router_sigmoid_parity: %s\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
