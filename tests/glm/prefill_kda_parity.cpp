// tests/glm/prefill_kda_parity.cpp - the KDA layer over a prompt in chunks (kda_conv_l2_prompt, kda_step_norm_prompt)
// against ggml-cpu at llama.cpp f1cee99, the graph of src/models/glm5-next.cpp build_kda_layer
// (G:686-762) over the whole prompt at once.
//
// For each seed (1..3) a prompt of 320 tokens with GLM's real dimensions (64 heads of 128, 3 x 8192 conv channels),
// random F32 projections, conv history, state and gate inputs, read three ways:
//   prompt path:  chunks of 1, 2, 61 and 256 tokens (a chunk shorter than the conv's 3 history taps, and a long one),
//                 the conv in place (h over the raw q|k|v, as GlmPrefill runs it), history and state carried
//   decode:       320 single-token steps (kda_conv_l2 / kda_step_norm with T = 1, the decode path)
//   oracle:       ggml_ssm_conv + silu + build_gdn_l2_norm, the G:707-721 gate, ggml_gated_delta_net in KDA mode
//                 (K = 1: the state after the last token), rms_norm * ssm_norm * sigmoid(g2): one graph of 320 tokens,
//                 llama.cpp's own sequential recurrence (its fused gated_delta_net; the chunked
//                 build_delta_net_chunking is not used by f1cee99 when the fused op is available)
// Checked: BITWISE the prompt path's conv outputs, y, final state and history against the decode steps (the same
// per-token arithmetic); against the oracle the conv outputs, y and the final state (tolerance 2e-5 of max |ref|,
// kda_parity's: F32 128-term dot products in another summation order; the recurrence over 320 tokens did not need
// more), and the history bitwise (data movement).  A negative control (one chunk's state not carried: the second
// chunk starts from zeros) must fail.  Prints the prompt kernels' time per token for 4,096-token chunks.
#include "strata/kernels/kda_decode.hpp"

#include "ggml_oracle.hpp"
#include "glm_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace glmtest;
namespace k = strata::kernels;

namespace {

constexpr int NH = 64, HD = 128, DI = NH * HD, C = 3 * DI;
constexpr int TOTAL = 320;
constexpr int CHUNKS[] = {1, 2, 61, 256};
constexpr double TOL = 2e-5;

struct Inputs {
    int T = 0;
    std::vector<float> qkv, hist, wq, wk, wv, fb, dtb, A, braw, gb, onw, state;
};

Inputs make_inputs(unsigned seed, int T) {
    std::mt19937 rng(seed * 104729u + (unsigned) T);
    Inputs in;
    in.T = T;
    in.qkv = normal(rng, (size_t) T * C, 1.0f);
    in.hist = normal(rng, (size_t) C * 3, 1.0f);
    in.wq = uniform(rng, (size_t) DI * 4, -0.6f, 0.6f);
    in.wk = uniform(rng, (size_t) DI * 4, -0.6f, 0.6f);
    in.wv = uniform(rng, (size_t) DI * 4, -0.6f, 0.6f);
    in.fb = normal(rng, (size_t) T * DI, 1.0f);
    in.dtb = normal(rng, DI, 0.5f);
    in.A = normal(rng, NH, 1.0f);
    for (auto& a : in.A) a = -std::exp(a);            // ssm_a = -exp(A_log), as conversion/glm.py stores it
    in.braw = normal(rng, (size_t) T * NH, 1.5f);
    in.gb = normal(rng, (size_t) T * DI, 1.5f);
    in.onw = normal(rng, HD, 0.2f, 1.0f);
    in.state = normal(rng, (size_t) NH * HD * HD, 0.3f);
    return in;
}

// Strata's state [h][i][j] <-> ggml's [h][j][i]
std::vector<float> transpose_heads(const std::vector<float>& s) {
    std::vector<float> o(s.size());
    for (int h = 0; h < NH; ++h)
        for (int i = 0; i < HD; ++i)
            for (int j = 0; j < HD; ++j) o[((size_t) h * HD + j) * HD + i] = s[((size_t) h * HD + i) * HD + j];
    return o;
}

struct OracleOut {
    std::vector<float> h, y, state;
};

// G:700-757 from the projections on, for the whole prompt as one ubatch (kda_parity's build_kda with K = 1)
OracleOut oracle(const Inputs& in, const k::KdaShape& sh) {
    Oracle o((size_t) 3 << 29);
    ggml_context* ctx = o.ctx;
    const int T = in.T;
    ggml_tensor* qkv = o.f32(in.qkv, C, T);
    const std::vector<float>* w[3] = {&in.wq, &in.wk, &in.wv};
    ggml_tensor* parts[3];
    for (int p = 0; p < 3; ++p) {
        std::vector<float> hp(in.hist.begin() + (size_t) p * DI * 3, in.hist.begin() + (size_t) (p + 1) * DI * 3);
        ggml_tensor* conv_state = o.f32(hp, 3, DI, 1);
        ggml_tensor* xp = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DI, T, qkv->nb[1], (size_t) p * DI * sizeof(float)));
        ggml_tensor* conv_x = ggml_concat(ctx, conv_state, ggml_transpose(ctx, ggml_reshape_3d(ctx, xp, DI, T, 1)), 0);
        ggml_tensor* X = ggml_ssm_conv(ctx, conv_x, o.f32(*w[p], 4, DI));
        X = ggml_silu(ctx, ggml_reshape_2d(ctx, X, DI, T));
        parts[p] = ggml_reshape_4d(ctx, X, HD, NH, T, 1);
    }
    ggml_tensor* g1 = ggml_add(ctx, o.f32(in.fb, DI, T), o.f32(in.dtb, DI));
    g1 = ggml_mul(ctx, ggml_reshape_3d(ctx, g1, HD, NH, T), ggml_reshape_3d(ctx, o.f32(in.A, NH), 1, NH, 1));
    g1 = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, g1, -1.0f)), sh.gate_lower_bound);
    g1 = ggml_reshape_4d(ctx, g1, HD, NH, T, 1);
    ggml_tensor* beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, o.f32(in.braw, NH, T), 1, NH, T, 1));
    ggml_tensor* state = o.f32(transpose_heads(in.state), HD, HD, NH, 1);
    const float n = (float) HD;
    auto l2 = [&](ggml_tensor* x) { return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f / n), 1.0f / sqrtf(n)); };
    ggml_tensor* q = l2(parts[0]);
    ggml_tensor* kk = l2(parts[1]);
    ggml_tensor* v = parts[2];
    ggml_tensor* gdn = ggml_gated_delta_net(ctx, q, kk, v, g1, beta, state, 1);
    ggml_tensor* out = ggml_cont(ctx, ggml_view_4d(ctx, gdn, HD, NH, T, 1, ggml_row_size(GGML_TYPE_F32, HD),
                                                   ggml_row_size(GGML_TYPE_F32, HD * NH),
                                                   ggml_row_size(GGML_TYPE_F32, HD * NH * T), 0));
    ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, out, HD, NH, T), sh.rms_eps),
                                   o.f32(in.onw, HD));
    ggml_tensor* gated =
        ggml_cont_2d(ctx, ggml_mul(ctx, normed, ggml_sigmoid(ctx, ggml_reshape_3d(ctx, o.f32(in.gb, DI, T), HD, NH, T))),
                     DI, T);
    q = ggml_cont(ctx, q);
    kk = ggml_cont(ctx, kk);
    v = ggml_cont(ctx, v);
    o.compute({q, kk, v, gated, gdn});
    OracleOut r;
    const auto qh = Oracle::read(q), kh = Oracle::read(kk), vh = Oracle::read(v);
    r.h.assign((size_t) T * C, 0.f);
    for (int t = 0; t < T; ++t) {
        std::copy(qh.begin() + (size_t) t * DI, qh.begin() + (size_t) (t + 1) * DI, r.h.begin() + (size_t) t * C);
        std::copy(kh.begin() + (size_t) t * DI, kh.begin() + (size_t) (t + 1) * DI, r.h.begin() + (size_t) t * C + DI);
        std::copy(vh.begin() + (size_t) t * DI, vh.begin() + (size_t) (t + 1) * DI, r.h.begin() + (size_t) t * C + 2 * DI);
    }
    r.y = Oracle::read(gated);
    const float* s = (const float*) gdn->data + (size_t) HD * NH * T;
    r.state = transpose_heads(std::vector<float>(s, s + (size_t) HD * HD * NH));
    return r;
}

struct Gpu {
    Dev<float> qkv, wq, wk, wv, fb, dtb, A, braw, gb, onw;
    explicit Gpu(const Inputs& in)
        : qkv(in.qkv), wq(in.wq), wk(in.wk), wv(in.wv), fb(in.fb), dtb(in.dtb), A(in.A), braw(in.braw), gb(in.gb),
          onw(in.onw) {}
};

// the prompt path: chunks in order, the conv in place over a copy of q|k|v; `carry` false resets the state and the
// history before the second chunk (the negative control)
void prompt_path(const Inputs& in, const Gpu& g, const k::KdaShape& sh, cudaStream_t st, Dev<float>& h, Dev<float>& y,
                 Dev<float>& hist, Dev<float>& state, bool carry) {
    GLM_CHECK(cudaMemcpy(h.p, g.qkv.p, (size_t) in.T * C * 4, cudaMemcpyDeviceToDevice));
    int t0 = 0, idx = 0;
    for (const int n : CHUNKS) {
        if (!carry && idx == 1) {
            hist.zero();
            state.zero();
        }
        k::kda_conv_l2_prompt(hist.p, h.p + (size_t) t0 * C, g.wq.p, g.wk.p, g.wv.p, h.p + (size_t) t0 * C, sh, n, st);
        k::kda_step_norm_prompt(state.p, h.p + (size_t) t0 * C, g.fb.p + (size_t) t0 * DI, g.dtb.p, g.A.p,
                                g.braw.p + (size_t) t0 * NH, g.gb.p + (size_t) t0 * DI, g.onw.p,
                                y.p + (size_t) t0 * DI, sh, n, st);
        t0 += n;
        ++idx;
    }
    GLM_CHECK(cudaStreamSynchronize(st));
}

bool run_seed(unsigned seed, const k::KdaShape& sh, cudaStream_t st) {
    std::printf("[prompt] seed %u, %d tokens in chunks of 1, 2, 61, 256\n", seed, TOTAL);
    const Inputs in = make_inputs(seed, TOTAL);
    const OracleOut ref = oracle(in, sh);
    Gpu g(in);
    bool ok = true;
    // the prompt path
    Dev<float> h((size_t) TOTAL * C), y((size_t) TOTAL * DI), hist(in.hist), state(in.state);
    prompt_path(in, g, sh, st, h, y, hist, state, true);
    const auto hp = h.down(), yp = y.down(), sp = state.down(), histp = hist.down();
    // the decode path, one token at a time
    Dev<float> hd((size_t) TOTAL * C), yd((size_t) TOTAL * DI), histd(in.hist), stated(in.state);
    for (int t = 0; t < TOTAL; ++t) {
        k::kda_conv_l2(histd.p, g.qkv.p + (size_t) t * C, g.wq.p, g.wk.p, g.wv.p, hd.p + (size_t) t * C, sh, 1, true, st);
        k::kda_step_norm(stated.p, hd.p + (size_t) t * C, g.fb.p + (size_t) t * DI, g.dtb.p, g.A.p,
                         g.braw.p + (size_t) t * NH, g.gb.p + (size_t) t * DI, g.onw.p, yd.p + (size_t) t * DI, sh, 1,
                         true, st);
    }
    GLM_CHECK(cudaStreamSynchronize(st));
    ok &= bitwise("prompt h == decode h", hp.data(), hd.down().data(), hp.size());
    ok &= bitwise("prompt y == decode y", yp.data(), yd.down().data(), yp.size());
    ok &= bitwise("prompt state == decode state", sp.data(), stated.down().data(), sp.size());
    ok &= bitwise("prompt history == decode history", histp.data(), histd.down().data(), histp.size());
    // against the oracle
    Err eq, ek, ev;
    for (int t = 0; t < TOTAL; ++t) {
        eq.merge(compare(hp.data() + (size_t) t * C, ref.h.data() + (size_t) t * C, DI));
        ek.merge(compare(hp.data() + (size_t) t * C + DI, ref.h.data() + (size_t) t * C + DI, DI));
        ev.merge(compare(hp.data() + (size_t) t * C + 2 * DI, ref.h.data() + (size_t) t * C + 2 * DI, DI));
    }
    ok &= report("conv+silu+l2 q (320 tokens)", eq, TOL);
    ok &= report("conv+silu+l2 k", ek, TOL);
    ok &= report("conv+silu v", ev, TOL);
    ok &= report("y = norm(o)*w*sigmoid(g2)", compare(yp.data(), ref.y.data(), yp.size()), TOL);
    ok &= report("state after token 320", compare(sp.data(), ref.state.data(), sp.size()), TOL);
    {
        std::vector<float> want(in.hist);
        for (int c = 0; c < C; ++c)
            for (int j = 0; j < 3; ++j) want[(size_t) c * 3 + j] = in.qkv[(size_t) (TOTAL - 3 + j) * C + c];
        ok &= bitwise("history after token 320", histp.data(), want.data(), want.size());
    }
    if (seed == 1) {   // the control: the state not carried from the first chunk to the second
        Dev<float> h2((size_t) TOTAL * C), y2((size_t) TOTAL * DI), hist2(in.hist), state2(in.state);
        prompt_path(in, g, sh, st, h2, y2, hist2, state2, false);
        const auto yy = y2.down();
        const Err e = compare(yy.data(), ref.y.data(), yy.size());
        const bool caught = e.rel() > 100 * TOL;
        std::printf("  %-34s rel %.3e against the oracle: %s\n", "control: state not carried", e.rel(),
                    caught ? "detected (PASS)" : "NOT detected (FAIL)");
        ok &= caught;
    }
    return ok;
}

void timing(const k::KdaShape& sh, cudaStream_t st) {
    constexpr int T = 4096;
    std::mt19937 rng(5);
    Dev<float> qkv(normal(rng, (size_t) T * C, 1.0f)), h((size_t) T * C), fb(normal(rng, (size_t) T * DI, 1.0f)),
        gb(normal(rng, (size_t) T * DI, 1.0f)), braw(normal(rng, (size_t) T * NH, 1.0f)), y((size_t) T * DI),
        w(uniform(rng, (size_t) DI * 4, -0.6f, 0.6f)), dtb(normal(rng, DI, 0.5f)), onw(normal(rng, HD, 0.2f, 1.0f)),
        hist((size_t) C * 3), state((size_t) NH * HD * HD);
    std::vector<float> a(NH, -1.0f);
    Dev<float> A(a);
    hist.zero();
    state.zero();
    cudaEvent_t e0, e1, e2;
    GLM_CHECK(cudaEventCreate(&e0));
    GLM_CHECK(cudaEventCreate(&e1));
    GLM_CHECK(cudaEventCreate(&e2));
    for (int r = 0; r < 2; ++r) {   // the first round warms the clocks
        GLM_CHECK(cudaEventRecord(e0, st));
        k::kda_conv_l2_prompt(hist.p, qkv.p, w.p, w.p, w.p, h.p, sh, T, st);
        GLM_CHECK(cudaEventRecord(e1, st));
        k::kda_step_norm_prompt(state.p, h.p, fb.p, dtb.p, A.p, braw.p, gb.p, onw.p, y.p, sh, T, st);
        GLM_CHECK(cudaEventRecord(e2, st));
        GLM_CHECK(cudaEventSynchronize(e2));
    }
    float c = 0, s = 0;
    GLM_CHECK(cudaEventElapsedTime(&c, e0, e1));
    GLM_CHECK(cudaEventElapsedTime(&s, e1, e2));
    std::printf("[timing] a 4,096-token chunk: conv %.2f ms (%.2f us/token), recurrence %.2f ms (%.2f us/token)\n", c,
                1000.0 * c / T, s, 1000.0 * s / T);
}

}  // namespace

int main() {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    k::KdaShape sh;
    sh.rms_eps = 9.999999747378752e-06f;   // glm5-next.attention.layer_norm_rms_epsilon in the GGUF
    bool ok = true;
    for (unsigned seed = 1; seed <= 3; ++seed) ok &= run_seed(seed, sh, st);
    timing(sh, st);
    std::printf("prefill_kda_parity: %s\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
