// tests/glm/kda_parity.cpp - the GLM-5.3-Flash KDA decode kernels (kda_decode.hpp) against ggml-cpu at llama.cpp
// f1cee99, the graph of src/models/glm5-next.cpp build_kda_layer (G:686-762) rebuilt op for op.
//
// Part 1, the kernels (seeds 1..3, windows T = 1..8, GLM's real dimensions: 64 heads of 128, 3 x 8192 conv channels):
// random F32 projections, conv history, state and gate inputs go through
//   GPU decode:  T single-token calls (conv with history update, recurrence with state write)
//   GPU window:  one verify call over T tokens (history/state read only), then for every n_keep in 1..T a commit
//   oracle:      ggml_ssm_conv + ggml_silu + build_gdn_l2_norm, the G:707-721 gate, ggml_gated_delta_net (KDA mode,
//                K = T snapshot slots: the state after every token), rms_norm * ssm_norm * sigmoid(g2)
// and the test checks: the conv outputs, y and the state after every token against the oracle (tolerance 2e-5 of
// the tensor's max |ref|: every value is a chain of F32 128-term dot products and exp/sqrt whose ggml-cpu SIMD
// summation order differs from the GPU's; one such product carries ~sqrt(128) * 6e-8 ~ 7e-7 relative rounding, and
// up to 8 recurrent steps plus the norms multiply that by ~10, leaving a 3x margin), and BITWISE: the window's y
// equals the single-token calls' y, a commit of n_keep equals n_keep single steps (state and conv history).
// A negative control runs the GPU with ssm_a exponentiated again (-exp(ssm_a)): ssm_a in the GGUF already is
// -exp(A_log) (conversion/glm.py:562-564) and G uses it as is (G:712-713); the control must fail the tolerance.
//
// Part 2, the layer (seed 7, T = 1, 4, 8): x -> Q8_0 projections on Strata's native GEMV (one Q8_1 quantization of
// x shared by q|k|v (one concatenated matrix), f_a, g_a and beta; f_b, g_b and W_o on their own inputs) -> the
// kernels -> W_o, against the oracle's mul_mat on the same Q8_0 bytes.  ggml-cpu quantizes the activations to Q8_0
// (x * (1/d)), the GPU to Q8_1 (x / d): the same integers except where x / d lands within an ulp of a rounding
// boundary, and the block products are summed in a different order, so this compares quantized paths at 1e-3 of
// max |ref|.  Measured: the projections themselves agree to ~3e-7; W_o's input y differs from the oracle's by ~1e-6,
// which moves about one of its 8192 values per token across a Q8 rounding boundary, and one such flip shifts every
// output by d * w ~ (3/127) * (1/sqrt(8192)) ~ 2.6e-4 - the size of the kda_out error at T = 1.
//
// Prints the per-kernel time for one token and for a T = 8 window (median of 100, events).
#include "strata/kernels/kda_decode.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml_oracle.hpp"
#include "glm_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace glmtest;
namespace k = strata::kernels;

namespace {

constexpr int NH = 64, HD = 128, DI = NH * HD, C = 3 * DI, NE = 4096;
constexpr double TOL_F32 = 2e-5, TOL_Q = 1e-3;

struct Inputs {
    int T = 0;
    std::vector<float> qkv, hist, wq, wk, wv, fb, dtb, A, braw, gb, onw, state;
};

Inputs make_inputs(unsigned seed, int T) {
    std::mt19937 rng(seed * 7919u + (unsigned) T);
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

// Strata's state [h][i][j] <-> ggml's [h][j][i] (s[(h*S + j)*S + i] = S[i][j])
std::vector<float> transpose_heads(const std::vector<float>& s) {
    std::vector<float> o(s.size());
    for (int h = 0; h < NH; ++h)
        for (int i = 0; i < HD; ++i)
            for (int j = 0; j < HD; ++j) o[((size_t) h * HD + j) * HD + i] = s[((size_t) h * HD + i) * HD + j];
    return o;
}

struct OracleOut {
    std::vector<float> h;                    // [T][C], q|k|v after conv/silu/l2, the kernel's layout
    std::vector<float> y;                    // [T][DI]
    std::vector<std::vector<float>> states;  // states[t] = after token t, Strata layout
    std::vector<float> out;                  // [T][NE] (layer test only)
};

// The KDA block of G from the projections on (G:700-757), for T tokens of one sequence.
struct KdaGraph {
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *gated = nullptr, *gdn = nullptr;
};

KdaGraph build_kda(Oracle& o, ggml_tensor* qkv /*[C, T]*/, const Inputs& in, ggml_tensor* fb /*[DI, T]*/,
                   ggml_tensor* braw /*[NH, T]*/, ggml_tensor* gb /*[DI, T]*/, float lb, float rms_eps) {
    ggml_context* ctx = o.ctx;
    const int T = in.T;
    const std::vector<float>* w[3] = {&in.wq, &in.wk, &in.wv};
    ggml_tensor* parts[3];
    for (int p = 0; p < 3; ++p) {
        std::vector<float> hp(in.hist.begin() + (size_t) p * DI * 3, in.hist.begin() + (size_t) (p + 1) * DI * 3);
        ggml_tensor* conv_state = o.f32(hp, 3, DI, 1);                              // [d_conv-1, d_inner, n_seqs]
        ggml_tensor* xp = ggml_cont(ctx, ggml_view_2d(ctx, qkv, DI, T, qkv->nb[1], (size_t) p * DI * sizeof(float)));
        ggml_tensor* x3 = ggml_reshape_3d(ctx, xp, DI, T, 1);
        ggml_tensor* conv_x = ggml_concat(ctx, conv_state, ggml_transpose(ctx, x3), 0);  // G:214
        ggml_tensor* cw = o.f32(*w[p], 4, DI);                                      // ggml_reshape_2d(conv_w, 4, DI)
        ggml_tensor* X = ggml_ssm_conv(ctx, conv_x, cw);
        X = ggml_reshape_2d(ctx, X, DI, T);
        X = ggml_silu(ctx, X);
        parts[p] = ggml_reshape_4d(ctx, X, HD, NH, T, 1);
    }
    // G:707-721, the lower-bound branch
    ggml_tensor* g1 = ggml_add(ctx, fb, o.f32(in.dtb, DI));
    ggml_tensor* A = ggml_reshape_3d(ctx, o.f32(in.A, NH), 1, NH, 1);
    g1 = ggml_reshape_3d(ctx, g1, HD, NH, T);
    g1 = ggml_mul(ctx, g1, A);
    g1 = ggml_sigmoid(ctx, ggml_scale(ctx, g1, -1.0f));
    g1 = ggml_scale(ctx, g1, lb);
    g1 = ggml_reshape_4d(ctx, g1, HD, NH, T, 1);
    ggml_tensor* beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, braw, 1, NH, T, 1));
    ggml_tensor* state = o.f32(transpose_heads(in.state), HD, HD, NH, 1);
    // build_gdn_l2_norm (models.h:14-18) with G's l2_eps = 1e-6
    const float n = (float) HD;
    auto l2 = [&](ggml_tensor* x) { return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-6f / n), 1.0f / sqrtf(n)); };
    KdaGraph G;
    G.q = l2(parts[0]);
    G.k = l2(parts[1]);
    G.v = parts[2];
    G.gdn = ggml_gated_delta_net(ctx, G.q, G.k, G.v, g1, beta, state, T);
    ggml_tensor* out = ggml_view_4d(ctx, G.gdn, HD, NH, T, 1, ggml_row_size(GGML_TYPE_F32, HD),
                                    ggml_row_size(GGML_TYPE_F32, HD * NH), ggml_row_size(GGML_TYPE_F32, HD * NH * T), 0);
    out = ggml_cont(ctx, out);
    ggml_tensor* g2 = ggml_reshape_3d(ctx, gb, HD, NH, T);
    ggml_tensor* ov = ggml_reshape_3d(ctx, out, HD, NH, T);
    ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, ov, rms_eps), o.f32(in.onw, HD));   // build_norm RMS
    G.gated = ggml_cont_2d(ctx, ggml_mul(ctx, normed, ggml_sigmoid(ctx, g2)), DI, T);
    G.q = ggml_cont(ctx, G.q);
    G.k = ggml_cont(ctx, G.k);
    G.v = ggml_cont(ctx, G.v);
    return G;
}

void read_kda(const KdaGraph& G, int T, OracleOut& r) {
    const auto q = Oracle::read(G.q), kk = Oracle::read(G.k), v = Oracle::read(G.v);
    r.h.assign((size_t) T * C, 0.f);
    for (int t = 0; t < T; ++t) {
        std::copy(q.begin() + (size_t) t * DI, q.begin() + (size_t) (t + 1) * DI, r.h.begin() + (size_t) t * C);
        std::copy(kk.begin() + (size_t) t * DI, kk.begin() + (size_t) (t + 1) * DI, r.h.begin() + (size_t) t * C + DI);
        std::copy(v.begin() + (size_t) t * DI, v.begin() + (size_t) (t + 1) * DI, r.h.begin() + (size_t) t * C + 2 * DI);
    }
    r.y = Oracle::read(G.gated);
    const float* base = (const float*) G.gdn->data + (size_t) HD * NH * T;
    const size_t snap = (size_t) HD * HD * NH;
    r.states.resize(T);
    for (int t = 0; t < T; ++t) {                    // slot s = the state s tokens back from the last
        const float* s = base + (size_t) (T - 1 - t) * snap;
        r.states[t] = transpose_heads(std::vector<float>(s, s + snap));
    }
}

OracleOut oracle_kernels(const Inputs& in, const k::KdaShape& sh) {
    Oracle o((size_t) 768 << 20);
    const int T = in.T;
    ggml_tensor* qkv = o.f32(in.qkv, C, T);
    KdaGraph G = build_kda(o, qkv, in, o.f32(in.fb, DI, T), o.f32(in.braw, NH, T), o.f32(in.gb, DI, T),
                           sh.gate_lower_bound, sh.rms_eps);
    o.compute({G.q, G.k, G.v, G.gated, G.gdn});
    OracleOut r;
    read_kda(G, T, r);
    return r;
}

struct GpuIn {
    Dev<float> qkv, wq, wk, wv, fb, dtb, A, braw, gb, onw;
    explicit GpuIn(const Inputs& in)
        : qkv(in.qkv), wq(in.wq), wk(in.wk), wv(in.wv), fb(in.fb), dtb(in.dtb), A(in.A), braw(in.braw), gb(in.gb),
          onw(in.onw) {}
};

bool part1(unsigned seed, int T, const k::KdaShape& sh, cudaStream_t st, bool negative_control) {
    std::printf("[kernels] seed %u, T = %d\n", seed, T);
    const Inputs in = make_inputs(seed, T);
    const OracleOut ref = oracle_kernels(in, sh);
    GpuIn g(in);
    bool ok = true;

    // ---- decode: T single-token calls
    Dev<float> hist(in.hist), state(in.state), h((size_t) T * C), y((size_t) T * DI);
    std::vector<std::vector<float>> dec_states, dec_hist;
    for (int t = 0; t < T; ++t) {
        k::kda_conv_l2(hist.p, g.qkv.p + (size_t) t * C, g.wq.p, g.wk.p, g.wv.p, h.p + (size_t) t * C, sh, 1, true, st);
        k::kda_step_norm(state.p, h.p + (size_t) t * C, g.fb.p + (size_t) t * DI, g.dtb.p, g.A.p,
                         g.braw.p + (size_t) t * NH, g.gb.p + (size_t) t * DI, g.onw.p, y.p + (size_t) t * DI, sh, 1,
                         true, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        dec_states.push_back(state.down());
        dec_hist.push_back(hist.down());
    }
    const auto h_dec = h.down(), y_dec = y.down();
    Err eq, ek, ev, ey, es;
    for (int t = 0; t < T; ++t) {
        eq.merge(compare(h_dec.data() + (size_t) t * C, ref.h.data() + (size_t) t * C, DI));
        ek.merge(compare(h_dec.data() + (size_t) t * C + DI, ref.h.data() + (size_t) t * C + DI, DI));
        ev.merge(compare(h_dec.data() + (size_t) t * C + 2 * DI, ref.h.data() + (size_t) t * C + 2 * DI, DI));
        es.merge(compare(dec_states[t].data(), ref.states[t].data(), dec_states[t].size()));
    }
    ey = compare(y_dec.data(), ref.y.data(), y_dec.size());
    ok &= report("conv+silu+l2 q", eq, TOL_F32);
    ok &= report("conv+silu+l2 k", ek, TOL_F32);
    ok &= report("conv+silu v", ev, TOL_F32);
    ok &= report("y = norm(o)*w*sigmoid(g2)", ey, TOL_F32);
    ok &= report("state after each token", es, TOL_F32);
    // the conv history is data movement: the last 3 inputs of [history | qkv_0 .. qkv_{T-1}]
    {
        std::vector<float> want(in.hist);
        for (int c = 0; c < C; ++c) {
            std::vector<float> seq = {in.hist[(size_t) c * 3], in.hist[(size_t) c * 3 + 1], in.hist[(size_t) c * 3 + 2]};
            for (int t = 0; t < T; ++t) seq.push_back(in.qkv[(size_t) t * C + c]);
            for (int j = 0; j < 3; ++j) want[(size_t) c * 3 + j] = seq[seq.size() - 3 + j];
        }
        ok &= bitwise("conv history after T", dec_hist.back().data(), want.data(), want.size());
    }

    // ---- window: verify over T tokens, then a commit per n_keep
    Dev<float> hist_w(in.hist), state_w(in.state), h_w((size_t) T * C), y_w((size_t) T * DI);
    k::kda_conv_l2(hist_w.p, g.qkv.p, g.wq.p, g.wk.p, g.wv.p, h_w.p, sh, T, false, st);
    k::kda_step_norm(state_w.p, h_w.p, g.fb.p, g.dtb.p, g.A.p, g.braw.p, g.gb.p, g.onw.p, y_w.p, sh, T, false, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    ok &= bitwise("window y == single-token y", y_w.down().data(), y_dec.data(), y_dec.size());
    ok &= bitwise("window h == single-token h", h_w.down().data(), h_dec.data(), h_dec.size());
    ok &= bitwise("verify leaves state untouched", state_w.down().data(), in.state.data(), in.state.size());
    ok &= bitwise("verify leaves history untouched", hist_w.down().data(), in.hist.data(), in.hist.size());
    bool commits = true;
    Dev<int32_t> nk(1);
    for (int n = 1; n <= T; ++n) {
        state_w.up(in.state);
        hist_w.up(in.hist);
        nk.up(std::vector<int32_t>{n});
        k::kda_conv_commit(hist_w.p, g.qkv.p, sh, nk.p, st);
        k::kda_step_commit(state_w.p, h_w.p, g.fb.p, g.dtb.p, g.A.p, g.braw.p, sh, nk.p, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        const auto s = state_w.down(), hh = hist_w.down();
        commits &= std::memcmp(s.data(), dec_states[n - 1].data(), s.size() * 4) == 0;
        commits &= std::memcmp(hh.data(), dec_hist[n - 1].data(), hh.size() * 4) == 0;
    }
    std::printf("  %-34s %s (n_keep 1..%d: state and history bitwise the single steps')\n", "commit(n_keep) replay",
                commits ? "PASS" : "FAIL", T);
    ok &= commits;

    if (negative_control) {
        std::vector<float> A2(in.A);
        for (auto& a : A2) a = -std::exp(a);         // exponentiated a second time
        Dev<float> A2d(A2), st2(in.state), y2((size_t) T * DI);
        k::kda_step_norm(st2.p, h_w.p, g.fb.p, g.dtb.p, A2d.p, g.braw.p, g.gb.p, g.onw.p, y2.p, sh, T, true, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        const auto yy = y2.down();
        const Err e = compare(yy.data(), ref.y.data(), yy.size());
        const bool caught = e.rel() > 100 * TOL_F32;
        std::printf("  %-34s rel %.3e against the oracle: %s\n", "control: ssm_a exponentiated again", e.rel(),
                    caught ? "detected (PASS)" : "NOT detected (FAIL)");
        ok &= caught;
    }
    return ok;
}

// ---- part 2: the whole layer from x, projections on the native Q8_0 GEMV
struct QW {
    std::vector<uint8_t> q;
    int n_in, rows;
};
QW qweight(std::mt19937& rng, int n_in, int rows) {
    const auto w = normal(rng, (size_t) n_in * rows, 1.0f / std::sqrt((float) n_in));
    return {quantize(GGML_TYPE_Q8_0, w, n_in, rows), n_in, rows};
}

bool part2(const k::KdaShape& sh, cudaStream_t st, bool timing) {
    std::mt19937 rng(7);
    const QW wq = qweight(rng, NE, DI), wk = qweight(rng, NE, DI), wv = qweight(rng, NE, DI);
    const QW fa = qweight(rng, NE, HD), fbw = qweight(rng, HD, DI), wb = qweight(rng, NE, NH);
    const QW ga = qweight(rng, NE, HD), gbw = qweight(rng, HD, DI), wo = qweight(rng, DI, NE);
    std::vector<uint8_t> wqkv(wq.q);
    wqkv.insert(wqkv.end(), wk.q.begin(), wk.q.end());
    wqkv.insert(wqkv.end(), wv.q.begin(), wv.q.end());
    Dev<uint8_t> d_qkv(wqkv), d_fa(fa.q), d_fb(fbw.q), d_b(wb.q), d_ga(ga.q), d_gb(gbw.q), d_wo(wo.q);
    bool ok = true;
    for (int T : {1, 4, 8}) {
        std::printf("[layer] seed 7, T = %d (x -> Q8_0 projections -> KDA -> W_o)\n", T);
        Inputs in = make_inputs(7, T);
        const auto x = normal(rng, (size_t) T * NE, 1.0f);
        // oracle
        Oracle o((size_t) 1024 << 20);
        ggml_tensor* X = o.f32(x, NE, T);
        ggml_tensor* Wq = o.weight(GGML_TYPE_Q8_0, wq.q, NE, DI);
        ggml_tensor* Wk = o.weight(GGML_TYPE_Q8_0, wk.q, NE, DI);
        ggml_tensor* Wv = o.weight(GGML_TYPE_Q8_0, wv.q, NE, DI);
        ggml_tensor* qkv = ggml_concat(o.ctx, ggml_concat(o.ctx, ggml_mul_mat(o.ctx, Wq, X), ggml_mul_mat(o.ctx, Wk, X), 0),
                                       ggml_mul_mat(o.ctx, Wv, X), 0);                       // [C, T]
        ggml_tensor* fA = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, fa.q, NE, HD), X);
        ggml_tensor* fB = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, fbw.q, HD, DI), fA);
        ggml_tensor* bR = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, wb.q, NE, NH), X);
        ggml_tensor* gA = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, ga.q, NE, HD), X);
        ggml_tensor* gB = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, gbw.q, HD, DI), gA);
        KdaGraph G = build_kda(o, qkv, in, fB, bR, gB, sh.gate_lower_bound, sh.rms_eps);
        ggml_tensor* out = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, wo.q, DI, NE), G.gated);
        o.compute({G.gated, out, G.gdn});
        const auto y_ref = Oracle::read(G.gated), out_ref = Oracle::read(out);

        // GPU
        Dev<float> dx(x), hist(in.hist), state(in.state), conv_wq(in.wq), conv_wk(in.wk), conv_wv(in.wv), dtb(in.dtb),
            A(in.A), onw(in.onw);
        Dev<float> qkv_d((size_t) T * C), fa_d((size_t) T * HD), ga_d((size_t) T * HD), b_d((size_t) T * NH),
            fb_d((size_t) T * DI), gb_d((size_t) T * DI), h((size_t) T * C), y((size_t) T * DI), out_d((size_t) T * NE);
        Dev<uint8_t> xq(k::native_q8_1_bytes(NE, T)), sq(k::native_q8_1_bytes(DI, T));
        auto run = [&](int nt, bool write) {
            k::native_quantize_q8_1(dx.p, xq.p, NE, nt, st);
            k::native_q8_0_mmvq(d_qkv.p, xq.p, qkv_d.p, NE, C, nt, st);
            k::native_q8_0_mmvq(d_fa.p, xq.p, fa_d.p, NE, HD, nt, st);
            k::native_q8_0_mmvq(d_ga.p, xq.p, ga_d.p, NE, HD, nt, st);
            k::native_q8_0_mmvq(d_b.p, xq.p, b_d.p, NE, NH, nt, st);
            k::native_q8_0_f32(d_fb.p, fa_d.p, sq.p, fb_d.p, HD, DI, nt, st);
            k::native_q8_0_f32(d_gb.p, ga_d.p, sq.p, gb_d.p, HD, DI, nt, st);
            k::kda_conv_l2(hist.p, qkv_d.p, conv_wq.p, conv_wk.p, conv_wv.p, h.p, sh, nt, write, st);
            k::kda_step_norm(state.p, h.p, fb_d.p, dtb.p, A.p, b_d.p, gb_d.p, onw.p, y.p, sh, nt, write, st);
            k::native_q8_0_f32(d_wo.p, y.p, sq.p, out_d.p, DI, NE, nt, st);
        };
        run(T, false);
        GLM_CHECK(cudaStreamSynchronize(st));
        const auto yv = y.down(), ov = out_d.down();
        ok &= report("y (quantized projections)", compare(yv.data(), y_ref.data(), yv.size()), TOL_Q);
        ok &= report("kda_out = W_o y", compare(ov.data(), out_ref.data(), ov.size()), TOL_Q);
        if (timing && T == 1) {
            std::printf("  one-token layer cost on this GPU (median of 100):\n");
            std::printf("    quantize x + q|k|v GEMV (4096 -> 24576 Q8_0)  %8.2f us\n", median_us([&] {
                k::native_quantize_q8_1(dx.p, xq.p, NE, 1, st);
                k::native_q8_0_mmvq(d_qkv.p, xq.p, qkv_d.p, NE, C, 1, st); }, st));
            std::printf("    f_a + g_a + beta GEMVs (4096 -> 128/128/64)   %8.2f us\n", median_us([&] {
                k::native_q8_0_mmvq(d_fa.p, xq.p, fa_d.p, NE, HD, 1, st);
                k::native_q8_0_mmvq(d_ga.p, xq.p, ga_d.p, NE, HD, 1, st);
                k::native_q8_0_mmvq(d_b.p, xq.p, b_d.p, NE, NH, 1, st); }, st));
            std::printf("    f_b + g_b GEMVs (128 -> 8192, with quantize)  %8.2f us\n", median_us([&] {
                k::native_q8_0_f32(d_fb.p, fa_d.p, sq.p, fb_d.p, HD, DI, 1, st);
                k::native_q8_0_f32(d_gb.p, ga_d.p, sq.p, gb_d.p, HD, DI, 1, st); }, st));
            std::printf("    W_o GEMV (8192 -> 4096, with quantize)        %8.2f us\n", median_us([&] {
                k::native_q8_0_f32(d_wo.p, y.p, sq.p, out_d.p, DI, NE, 1, st); }, st));
            std::printf("    whole layer (all of the above + the kernels)  %8.2f us\n", median_us([&] { run(1, false); }, st));
        }
    }
    return ok;
}

void timings(const k::KdaShape& sh, cudaStream_t st) {
    const Inputs in = make_inputs(99, 8);
    GpuIn g(in);
    Dev<float> hist(in.hist), state(in.state), h((size_t) 8 * C), y((size_t) 8 * DI);
    Dev<int32_t> nk(std::vector<int32_t>{8});
    std::printf("[timing] median of 100 on this GPU\n");
    std::printf("  kda_conv_l2      1 token  (decode)   %8.2f us\n",
                median_us([&] { k::kda_conv_l2(hist.p, g.qkv.p, g.wq.p, g.wk.p, g.wv.p, h.p, sh, 1, true, st); }, st));
    std::printf("  kda_step_norm    1 token  (decode)   %8.2f us\n", median_us([&] {
        k::kda_step_norm(state.p, h.p, g.fb.p, g.dtb.p, g.A.p, g.braw.p, g.gb.p, g.onw.p, y.p, sh, 1, true, st); }, st));
    std::printf("  kda_conv_l2      8 tokens (verify)   %8.2f us\n",
                median_us([&] { k::kda_conv_l2(hist.p, g.qkv.p, g.wq.p, g.wk.p, g.wv.p, h.p, sh, 8, false, st); }, st));
    std::printf("  kda_step_norm    8 tokens (verify)   %8.2f us\n", median_us([&] {
        k::kda_step_norm(state.p, h.p, g.fb.p, g.dtb.p, g.A.p, g.braw.p, g.gb.p, g.onw.p, y.p, sh, 8, false, st); }, st));
    std::printf("  kda_conv_commit  n_keep 8            %8.2f us\n",
                median_us([&] { k::kda_conv_commit(hist.p, g.qkv.p, sh, nk.p, st); }, st));
    std::printf("  kda_step_commit  n_keep 8            %8.2f us\n", median_us([&] {
        k::kda_step_commit(state.p, h.p, g.fb.p, g.dtb.p, g.A.p, g.braw.p, sh, nk.p, st); }, st));
}

}  // namespace

int main(int argc, char** argv) {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    k::KdaShape sh;                                  // GLM-5.3-Flash: 64 x 128, lb -5, rms eps 1e-5, l2 eps 1e-6
    sh.rms_eps = 9.999999747378752e-06f;             // glm5-next.attention.layer_norm_rms_epsilon in the GGUF
    bool ok = true;
    for (unsigned seed = 1; seed <= 3; ++seed)
        for (int T = 1; T <= 8; ++T) ok &= part1(seed, T, sh, st, seed == 1 && T == 4);
    if (!has_flag(argc, argv, "--kernels-only")) ok &= part2(sh, st, true);
    timings(sh, st);
    std::printf("kda_parity: %s\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
