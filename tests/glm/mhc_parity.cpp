// tests/glm/mhc_parity.cpp - the GLM-5.3-Flash mHC kernels (mhc.hpp) against ggml-cpu at llama.cpp f1cee99, the
// graph of src/models/glm5-next.cpp build_hc_pre / build_hc_post / build_hc_mean (G:370-547) with the fused
// ggml_dsv4_hc_* ops llama.cpp uses by default; the unfused Sinkhorn graph (build_hc_sinkhorn, G:417-452) is a second
// oracle for comb.  Real dimensions (4 streams of 4096, 24 mixing values, 20 Sinkhorn iterations, eps 1e-6), seeds
// 1..3, T = 1, 3, 8.
//
// Checks and tolerances (of max |ref|):
//   flat RMS norm                        1e-6   one F32 sum of 16384 squares (ggml sums in double) and one sqrt
//   mhc_pre from the ORACLE's mixes      2e-6   pre / post: a sigmoid; comb: a softmax and 39 normalizations, each a
//     (pre, post, comb, cur)                    4-term F32 sum and a reciprocal: ~40 roundings of 6e-8, x ~1 margin
//   comb against the unfused graph       2e-6   (the graph divides where the fused op multiplies by a reciprocal)
//   mhc_post from the oracle's post/comb 1e-6   5 products and 4 adds per value
//   mhc_mean                             bitwise (three adds in stream order and an exact * 0.25)
//   mixes and cur through the native Q8_0 GEMV (hc_fn . flat): 1e-3, quantized paths (see kda_parity's part 2)
// plus the Sinkhorn output's own property (the last normalization's sums, over src, are 1 within 1e-5).
#include "strata/kernels/mhc.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml_oracle.hpp"
#include "glm_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace glmtest;
namespace k = strata::kernels;

namespace {

constexpr int NE = 4096, HC = 4, MIX = 24, FLAT = HC * NE;

ggml_tensor* view1(ggml_context* ctx, ggml_tensor* t, int64_t n, int64_t i0) {
    return ggml_view_1d(ctx, t, n, ggml_row_size(t->type, i0));
}
ggml_tensor* view2(ggml_context* ctx, ggml_tensor* t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

// G:417-452, the unfused graph
ggml_tensor* sinkhorn_graph(ggml_context* ctx, ggml_tensor* comb, float eps_v, int iters) {
    comb = ggml_soft_max(ctx, comb);
    ggml_tensor* eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx, eps, eps_v);
    comb = ggml_add(ctx, comb, eps);
    auto norm_cols = [&]() {
        ggml_tensor* csd = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));
        ggml_tensor* col_sum = ggml_sum_rows(ctx, csd);
        col_sum = ggml_add(ctx, col_sum, eps);
        col_sum = ggml_permute(ctx, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx, comb, col_sum);
    };
    auto norm_rows = [&]() {
        ggml_tensor* row_sum = ggml_sum_rows(ctx, comb);
        row_sum = ggml_add(ctx, row_sum, eps);
        comb = ggml_div(ctx, comb, row_sum);
    };
    norm_cols();
    for (int i = 1; i < iters; ++i) {
        norm_rows();
        norm_cols();
    }
    return comb;
}

bool run(unsigned seed, int T, const k::MhcShape& sh, cudaStream_t st, bool timing) {
    std::printf("[mhc] seed %u, T = %d\n", seed, T);
    std::mt19937 rng(seed * 104729u + (unsigned) T);
    const auto x = normal(rng, (size_t) T * FLAT, 1.0f);
    const auto f = normal(rng, (size_t) T * NE, 1.0f);
    const auto fn = normal(rng, (size_t) FLAT * MIX, 2.0f / std::sqrt((float) FLAT));
    const auto qfn = quantize(GGML_TYPE_Q8_0, fn, FLAT, MIX);
    const auto scale = uniform(rng, 3, 0.5f, 2.0f);
    const auto base = normal(rng, MIX, 1.0f);

    // ---- oracle (G:454-510 with the fused ops, then build_hc_post and build_hc_mean)
    Oracle o((size_t) 256 << 20);
    ggml_context* ctx = o.ctx;
    ggml_tensor* X = o.f32(x, NE, HC, T);
    ggml_tensor* S = o.f32(scale, 3);
    ggml_tensor* B = o.f32(base, MIX);
    ggml_tensor* flat = ggml_rms_norm(ctx, ggml_reshape_2d(ctx, X, FLAT, T), sh.rms_eps);
    ggml_tensor* mixes = ggml_mul_mat(ctx, o.weight(GGML_TYPE_Q8_0, qfn, FLAT, MIX), flat);
    ggml_tensor* pre = view2(ctx, mixes, HC, T, 0);
    pre = ggml_add(ctx, ggml_mul(ctx, pre, view1(ctx, S, 1, 0)), view1(ctx, B, HC, 0));
    pre = ggml_scale_bias(ctx, ggml_sigmoid(ctx, pre), 1.0f, sh.hc_eps);
    ggml_tensor* post = view2(ctx, mixes, HC, T, HC);
    post = ggml_add(ctx, ggml_mul(ctx, post, view1(ctx, S, 1, 1)), view1(ctx, B, HC, HC));
    post = ggml_scale(ctx, ggml_sigmoid(ctx, post), 2.0f);
    ggml_tensor* comb = ggml_dsv4_hc_comb(ctx, mixes, S, B, sh.hc_eps, sh.sinkhorn_iters);
    ggml_tensor* comb_u = view2(ctx, mixes, HC * HC, T, 2 * HC);
    comb_u = ggml_add(ctx, ggml_mul(ctx, comb_u, view1(ctx, S, 1, 2)), view1(ctx, B, HC * HC, 2 * HC));
    comb_u = sinkhorn_graph(ctx, ggml_reshape_3d(ctx, comb_u, HC, HC, T), sh.hc_eps, sh.sinkhorn_iters);
    ggml_tensor* cur = ggml_dsv4_hc_pre(ctx, X, pre);
    ggml_tensor* F = o.f32(f, NE, T);
    ggml_tensor* outp = ggml_dsv4_hc_post(ctx, F, X, post, comb);
    // build_hc_mean on the post output
    ggml_tensor* acc = ggml_view_2d(ctx, outp, NE, T, outp->nb[2], 0);
    for (int s = 1; s < HC; ++s) acc = ggml_add(ctx, acc, ggml_view_2d(ctx, outp, NE, T, outp->nb[2], s * outp->nb[1]));
    ggml_tensor* head = ggml_scale(ctx, acc, 1.0f / HC);
    pre = ggml_cont(ctx, pre);
    post = ggml_cont(ctx, post);
    comb_u = ggml_cont(ctx, comb_u);
    o.compute({flat, mixes, pre, post, comb, comb_u, cur, outp, head});
    const auto r_flat = Oracle::read(flat), r_mix = Oracle::read(mixes), r_pre = Oracle::read(pre),
               r_post = Oracle::read(post), r_comb = Oracle::read(comb), r_combu = Oracle::read(comb_u),
               r_cur = Oracle::read(cur), r_out = Oracle::read(outp), r_head = Oracle::read(head);

    // ---- GPU
    Dev<float> dx(x), df(f), dS(scale), dB(base), dflat((size_t) T * FLAT), dmix((size_t) T * MIX),
        dmix_ref(r_mix), dcur((size_t) T * NE), dpost((size_t) T * HC), dcomb((size_t) T * HC * HC),
        dpost_ref(r_post), dcomb_ref(r_comb), dout((size_t) T * FLAT), dhead((size_t) T * NE);
    Dev<uint8_t> dfn(qfn), dq(k::native_q8_1_bytes(FLAT, T));
    bool ok = true;
    k::mhc_flat_norm(dx.p, dflat.p, sh, T, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    const auto g_flat = dflat.down();
    ok &= report("flat rms_norm", compare(g_flat.data(), r_flat.data(), g_flat.size()), 1e-6);

    // mhc_pre on the oracle's mixes: the kernel alone
    k::mhc_pre(dx.p, dmix_ref.p, dS.p, dB.p, dcur.p, dpost.p, dcomb.p, sh, T, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    const auto g_cur = dcur.down(), g_post = dpost.down(), g_comb = dcomb.down();
    // pre is not an output of its own: cur = sum_h x[h] * pre[h] carries it
    ok &= report("post = 2 sigmoid(.)", compare(g_post.data(), r_post.data(), g_post.size()), 2e-6);
    ok &= report("comb (fused-op oracle)", compare(g_comb.data(), r_comb.data(), g_comb.size()), 2e-6);
    ok &= report("comb (unfused graph G:417-452)", compare(g_comb.data(), r_combu.data(), g_comb.size()), 2e-6);
    ok &= report("cur = sum_h x[h] pre[h]", compare(g_cur.data(), r_cur.data(), g_cur.size()), 2e-6);
    {
        // The last Sinkhorn step normalizes over src for each dst, so those sums are 1 up to eps; the sums over dst
        // are not after 20 iterations (Sinkhorn has not converged there; the oracle agrees), printed for information.
        double worst_src = 0, worst_dst = 0;
        for (int t = 0; t < T; ++t)
            for (int a = 0; a < HC; ++a) {
                double over_src = 0, over_dst = 0;
                for (int b = 0; b < HC; ++b) {
                    over_src += g_comb[(size_t) t * 16 + a + HC * b];   // dst a, every src
                    over_dst += g_comb[(size_t) t * 16 + b + HC * a];   // src a, every dst
                }
                worst_src = std::max(worst_src, std::fabs(over_src - 1));
                worst_dst = std::max(worst_dst, std::fabs(over_dst - 1));
            }
        const bool p = worst_src <= 1e-5;
        std::printf("  %-34s max |sum over src - 1| %.3e (tol 1e-5)  %s; over dst %.3e (not normalized last)\n",
                    "comb after the last normalization", worst_src, p ? "PASS" : "FAIL", worst_dst);
        ok &= p;
    }
    // mhc_post on the oracle's post/comb, out of place and in place
    k::mhc_post(df.p, dx.p, dpost_ref.p, dcomb_ref.p, dout.p, sh, T, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    const auto g_out = dout.down();
    ok &= report("mhc_post", compare(g_out.data(), r_out.data(), g_out.size()), 1e-6);
    {
        Dev<float> inplace(x);
        k::mhc_post(df.p, inplace.p, dpost_ref.p, dcomb_ref.p, inplace.p, sh, T, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        ok &= bitwise("mhc_post in place == out of place", inplace.down().data(), g_out.data(), g_out.size());
    }
    Dev<float> dout_ref(r_out);
    k::mhc_mean(dout_ref.p, dhead.p, sh, T, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    ok &= bitwise("mhc_mean == build_hc_mean", dhead.down().data(), r_head.data(), r_head.size());

    // the whole pre phase with the GEMV: flat norm -> native Q8_0 GEMV -> mhc_pre
    auto pre_phase = [&] {
        k::mhc_flat_norm(dx.p, dflat.p, sh, T, st);
        k::native_quantize_q8_1(dflat.p, dq.p, FLAT, T, st);
        k::native_q8_0_mmvq(dfn.p, dq.p, dmix.p, FLAT, MIX, T, st);
        k::mhc_pre(dx.p, dmix.p, dS.p, dB.p, dcur.p, dpost.p, dcomb.p, sh, T, st);
    };
    pre_phase();
    GLM_CHECK(cudaStreamSynchronize(st));
    const auto g_mix = dmix.down(), g_cur2 = dcur.down();
    ok &= report("mixes = hc_fn . flat (Q8_0 GEMV)", compare(g_mix.data(), r_mix.data(), g_mix.size()), 1e-3);
    ok &= report("cur through the GEMV", compare(g_cur2.data(), r_cur.data(), g_cur2.size()), 1e-3);

    if (timing) {
        std::printf("  [timing] median of 100, T = %d\n", T);
        std::printf("    mhc_flat_norm                       %8.2f us\n",
                    median_us([&] { k::mhc_flat_norm(dx.p, dflat.p, sh, T, st); }, st));
        std::printf("    quantize + hc_fn GEMV (16384 -> 24) %8.2f us\n", median_us([&] {
            k::native_quantize_q8_1(dflat.p, dq.p, FLAT, T, st);
            k::native_q8_0_mmvq(dfn.p, dq.p, dmix.p, FLAT, MIX, T, st); }, st));
        std::printf("    mhc_pre                             %8.2f us\n",
                    median_us([&] { k::mhc_pre(dx.p, dmix.p, dS.p, dB.p, dcur.p, dpost.p, dcomb.p, sh, T, st); }, st));
        std::printf("    mhc_post                            %8.2f us\n",
                    median_us([&] { k::mhc_post(df.p, dx.p, dpost.p, dcomb.p, dout.p, sh, T, st); }, st));
        std::printf("    mhc_mean                            %8.2f us\n",
                    median_us([&] { k::mhc_mean(dout.p, dhead.p, sh, T, st); }, st));
    }
    return ok;
}

}  // namespace

int main() {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    k::MhcShape sh;
    sh.rms_eps = 9.999999747378752e-06f;     // glm5-next.attention.layer_norm_rms_epsilon
    sh.hc_eps = 9.999999974752427e-07f;      // glm5-next.hyper_connection.epsilon
    bool ok = true;
    for (unsigned seed = 1; seed <= 3; ++seed)
        for (int T : {1, 3, 8}) ok &= run(seed, T, sh, st, seed == 1 && T == 1);
    std::printf("mhc_parity: %s\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
