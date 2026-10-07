// tests/glm/ffn_clamp_parity.cpp - GLM-5.3-Flash's dense SwiGLU FFNs with the clamp (ffn_clamp.hpp): the three leading
// dense layers (4096 -> 12288 -> 4096, Q8_0) and the MoE layers' shared expert (4096 -> 2048 -> 4096, Q8_0), against
// ggml-cpu at llama.cpp f1cee99: mul_mat, ggml_swiglu_clamp(gate, up, 10), mul_mat - build_ffn (llama-graph.cpp
// 1850-1857).  T = 1, 4, 8 tokens; weights scaled so ~30% of the gate/up values pass +-10.
//
// Checks and tolerances (of max |ref|):
//   gate, up, out against an exact reference: f64 sums of ggml's dequantized Q8_0 weights times the Q8_1 activation
//   the GPU used (x's, and h's for out)                                    1e-6 (F32 block sums in another order)
//   the same against ggml's mul_mat chain                                   5e-3: ggml-cpu's Q8_0 x Q8_0 mul_mat is
//     itself off the exact quantized product by up to 2e-4 at some shapes (measured here at 2048 rows x 8 tokens,
//     printed as "oracle vs exact"; 1e-7 at the others), and h's Q8 rounding amplifies a difference in gate/up 10x
//   the fused clamp + Q8_1 kernel against native_quantize_q8_1 of the host's ggml_swiglu_clamp element on the GPU's own
//   gate/up: the blocks' scales bitwise; a quantized value may differ by one step where expf's last bit (GPU libm vs
//   glibc) moves h across a rounding boundary - at most 1e-4 of the values, never more than one step
//   limit 1e30 equals the unclamped SwiGLU (limit 0) bitwise, and limit 10 changes the output.
#include "strata/kernels/ffn_clamp.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml_oracle.hpp"
#include "glm_test_util.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace glmtest;
namespace k = strata::kernels;

namespace {

constexpr int NE = 4096;
constexpr float LIMIT = 10.0f;

bool run(const char* what, int FF, int T, unsigned seed, cudaStream_t st, bool timing) {
    std::printf("[ffn] %s: %d -> %d -> %d, T = %d\n", what, NE, FF, NE, T);
    std::mt19937 rng(seed);
    const float sd = 9.6f / 64.0f;   // gate/up ~ N(0, 9.6)
    const auto qg = quantize(GGML_TYPE_Q8_0, normal(rng, (size_t) NE * FF, sd), NE, FF);
    const auto qu = quantize(GGML_TYPE_Q8_0, normal(rng, (size_t) NE * FF, sd), NE, FF);
    const auto qd = quantize(GGML_TYPE_Q8_0, normal(rng, (size_t) NE * FF, 1.0f / std::sqrt((float) FF)), FF, NE);
    const auto x = normal(rng, (size_t) T * NE, 1.0f);

    Oracle o((size_t) 512 << 20);
    ggml_tensor* X = o.f32(x, NE, T);
    ggml_tensor* up = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, qu, NE, FF), X);     // build_ffn: tmp = up(x)
    ggml_tensor* gate = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, qg, NE, FF), X);   // cur = gate(x) (LLM_FFN_PAR)
    ggml_tensor* h = ggml_swiglu_clamp(o.ctx, gate, up, LIMIT);                          // 1857
    ggml_tensor* out = ggml_mul_mat(o.ctx, o.weight(GGML_TYPE_Q8_0, qd, FF, NE), h);
    o.compute({gate, up, h, out});
    const auto r_gate = Oracle::read(gate), r_up = Oracle::read(up), r_out = Oracle::read(out);

    Dev<uint8_t> dg(qg), du(qu), dd(qd), scr(k::ffn_clamp_scratch_bytes(NE, FF, T));
    Dev<float> dx(x), dout((size_t) T * NE);
    k::FfnClampWeights w;
    w.gate = dg.p;
    w.up = du.p;
    w.down = dd.p;
    bool ok = true;
    auto run_gpu = [&](float limit) {
        k::ffn_clamp(w, dx.p, dout.p, NE, FF, T, limit, scr.p, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        return dout.down();
    };
    const auto out0 = run_gpu(0.0f), outh = run_gpu(1e30f);
    ok &= bitwise("limit 1e30 == unclamped", outh.data(), out0.data(), out0.size());
    const auto out10 = run_gpu(LIMIT);
    std::vector<float> g_gate((size_t) T * FF), g_up((size_t) T * FF);
    GLM_CHECK(cudaMemcpy(g_gate.data(), k::ffn_clamp_gate(scr.p, NE, FF, T), g_gate.size() * 4, cudaMemcpyDeviceToHost));
    GLM_CHECK(cudaMemcpy(g_up.data(), k::ffn_clamp_up(scr.p, NE, FF, T), g_up.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> g_hq(k::native_q8_1_bytes(FF, T));
    GLM_CHECK(cudaMemcpy(g_hq.data(), k::ffn_clamp_h_q8_1(scr.p, NE, FF, T), g_hq.size(), cudaMemcpyDeviceToHost));
    // the exact references on the GPU's own Q8_1 activations
    auto dq81 = [](const std::vector<uint8_t>& q) {
        std::vector<float> f(q.size() / 36 * 32);
        for (size_t b = 0; b < q.size() / 36; ++b) {
            ggml_fp16_t dh;
            std::memcpy(&dh, &q[b * 36], 2);
            const float d = ggml_fp16_to_fp32(dh);
            for (int i = 0; i < 32; ++i) f[b * 32 + i] = d * (float) (int8_t) q[b * 36 + 4 + i];
        }
        return f;
    };
    auto exact = [](const std::vector<uint8_t>& wq, int n_in, int rows, const std::vector<float>& a, int nt) {
        std::vector<float> wf((size_t) n_in * rows), y((size_t) nt * rows);
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(wq.data(), wf.data(), (int64_t) wf.size());
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int c = 0; c < n_in; ++c) acc += (double) wf[(size_t) r * n_in + c] * a[(size_t) t * n_in + c];
                y[(size_t) t * rows + r] = (float) acc;
            }
        return y;
    };
    std::vector<uint8_t> g_xq(k::native_q8_1_bytes(NE, T));
    GLM_CHECK(cudaMemcpy(g_xq.data(), scr.p, g_xq.size(), cudaMemcpyDeviceToHost));
    const auto xa = dq81(g_xq);
    const auto e_gate = exact(qg, NE, FF, xa, T), e_up = exact(qu, NE, FF, xa, T);
    const auto e_out = exact(qd, FF, NE, dq81(g_hq), T);
    ok &= report("gate (vs exact)", compare(g_gate.data(), e_gate.data(), e_gate.size()), 1e-6);
    ok &= report("up (vs exact)", compare(g_up.data(), e_up.data(), e_up.size()), 1e-6);
    ok &= report("gate (vs ggml mul_mat)", compare(g_gate.data(), r_gate.data(), r_gate.size()), 5e-3);
    ok &= report("up (vs ggml mul_mat)", compare(g_up.data(), r_up.data(), r_up.size()), 5e-3);
    {
        const Err oe = compare(r_gate.data(), e_gate.data(), e_gate.size());
        std::printf("  %-34s rel %.3e (information: ggml-cpu's own Q8_0 mul_mat at this shape)\n", "oracle vs exact (gate)",
                    oe.rel());
    }
    {
        // the host's ggml_swiglu_clamp element on the GPU's gate/up, quantized by native_quantize_q8_1
        std::vector<float> hh(g_gate.size());
        size_t clamped = 0;
        for (size_t i = 0; i < hh.size(); ++i) {
            float g = g_gate[i], u = g_up[i];
            clamped += g > LIMIT || std::fabs(u) > LIMIT;
            g = std::min(g, LIMIT);
            u = std::clamp(u, -LIMIT, LIMIT);
            hh[i] = g / (1.f + expf(-g)) * u;
        }
        Dev<float> dh(hh);
        Dev<uint8_t> ref(g_hq.size());
        k::native_quantize_q8_1(dh.p, ref.p, FF, T, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        const auto r = ref.down();
        size_t scale_diff = 0, q_diff = 0, q_big = 0;
        for (size_t b = 0; b < r.size() / 36; ++b) {
            scale_diff += std::memcmp(&r[b * 36], &g_hq[b * 36], 4) != 0;
            for (int i = 0; i < 32; ++i) {
                const int a = (int8_t) r[b * 36 + 4 + i], c = (int8_t) g_hq[b * 36 + 4 + i];
                q_diff += a != c;
                q_big += std::abs(a - c) > 1;
            }
        }
        const size_t n = r.size() / 36 * 32;
        const bool p = scale_diff == 0 && q_big == 0 && q_diff * 10000 <= n;
        std::printf("  %-34s %zu of %zu pairs clamped; scales differ in %zu blocks, %zu of %zu values by one step  %s\n",
                    "clamp + Q8_1 kernel", clamped, hh.size(), scale_diff, q_diff, n, p ? "PASS" : "FAIL");
        ok &= p;
    }
    ok &= report("out = W_down h (vs exact)", compare(out10.data(), e_out.data(), e_out.size()), 1e-6);
    ok &= report("out (vs ggml chain)", compare(out10.data(), r_out.data(), r_out.size()), 5e-3);
    {
        const Err moved = compare(out10.data(), out0.data(), out0.size());
        const bool p = moved.max_abs > 0;
        std::printf("  %-34s output moved by up to %.3e  %s\n", "clamp active", moved.max_abs, p ? "PASS" : "FAIL");
        ok &= p;
    }
    if (timing) {
        const double whole = median_us([&] { k::ffn_clamp(w, dx.p, dout.p, NE, FF, T, LIMIT, scr.p, st); }, st);
        const double fused = median_us([&] {
            k::native_swiglu_clamp_quantize_q8_1(k::ffn_clamp_gate(scr.p, NE, FF, T), k::ffn_clamp_up(scr.p, NE, FF, T),
                                                 LIMIT, const_cast<void*>(k::ffn_clamp_h_q8_1(scr.p, NE, FF, T)), FF, T,
                                                 st); }, st);
        std::printf("  [timing] T = %d: ffn_clamp %.2f us, of it the clamp + Q8_1 kernel %.2f us (median of 100, best "
                    "of 5; floor %.2f us)\n", T, whole, fused, floor_us(st));
    }
    return ok;
}

}  // namespace

int main() {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    bool ok = true;
    unsigned seed = 21;
    for (int T : {1, 4, 8}) ok &= run("dense layers 0-2", 12288, T, seed++, st, T == 1);
    for (int T : {1, 8}) ok &= run("shared expert", 2048, T, seed++, st, T == 1);
    std::printf("ffn_clamp_parity: %s\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
