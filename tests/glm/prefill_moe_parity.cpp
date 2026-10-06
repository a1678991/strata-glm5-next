// tests/glm/prefill_moe_parity.cpp - the prompt path's grouped routed experts (glm::expert_rows: per expert its rows
// gathered, gate|up and down dequantized to FP16, two GEMMs, ggml_swiglu_clamp between them, the weighted rows
// added; port plan Stage 6) against ggml-cpu at llama.cpp f1cee99: build_moe_ffn's expert half
// (src/llama-graph.cpp ~2200-2370: mul_mat_id gate and up, ggml_swiglu_clamp(gate, up, swiglu_clamp_exp),
// mul_mat_id down, times the routing weights, summed over the k slots), on random inputs.
//
// Cases: GLM-5.3-Flash UD-IQ4_XS's three expert formats - gate/up IQ3_S with down IQ4_XS (39 layers), IQ3_S / Q6_K
// (layers 12 and 44), IQ4_XS / Q6_K (layer 11) - at n_embd 4096 (the real width; the kernels read n_ff from the
// layout) with n_ff 512 (and 2048, the real one, for the first format), and UD-IQ3_XXS's two new ones at 4096 x 2048:
// IQ2_S / IQ3_S (39 layers, IQ3_S down rows) and IQ2_S / IQ4_XS (layers 12, 44; its layer 11 is IQ3_S / IQ4_XS); each
// also through native_expert_supported, the format gate GlmPrefill applies to every layer at startup; 8 experts, top-4 per token, 160 tokens, weights quantized
// by ggml's own quantizer into GGUF blocks laid out as Strata's native blob (gate rows | up rows | down rows).
// Activations and weights are scaled so that about 1-2% of the gate and up pre-activations pass the clamp limit 10.
// Three results: the GPU's, the oracle's, and an exact one (float64 over the weights as f1cee99's ggml dequantizes
// them, the clamp and the weighted sum).  The oracle quantizes its activations to Q8_K (ggml-cpu's vec_dot type for
// these formats: x before gate/up, h before down), the GPU rounds x, h and the dequantized weights to FP16, so both
// differ from the exact result by their own rounding.  Measured (relative to max |exact|): GPU ~6e-4, the oracle
// ~2e-2 (random data at Q8_K's 1/127-of-the-block-maximum step: ~0.7% per product, compounded through h).  Criteria:
// GPU vs exact <= 2e-3 (FP16: 2^-11 relative per rounded value), and GPU vs oracle no further than the oracle's own
// distance to the exact result plus that 2e-3 (the triangle inequality: the GPU adds nothing beyond its own FP16
// rounding).  The oracle's graph itself is checked against the exact result over F32 copies of the weights (1e-5:
// ggml-cpu then keeps the activations in f32).  A negative control (the clamp off: limit 0) must fail the GPU-vs-exact tolerance.  Prints the time of
// one expert over 512 rows at n_ff 2048.
#include "strata/prefill/glm_prefill_kernels.hpp"

#include "ggml_oracle.hpp"
#include "glm_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace glmtest;
namespace k = strata::kernels;
namespace pg = strata::prefill::glm;

namespace {

constexpr int E = 4096, NEXP = 8, KUSED = 4, T = 160;
constexpr float LIMIT = 10.0f;
constexpr double TOL_EXACT = 2e-3;

struct Case {
    ggml_type gu, d;
    int nff;
    const char* name;
};

struct Weights {
    std::vector<std::vector<uint8_t>> gate, up, down;   // per expert, GGUF blocks
    std::vector<std::vector<float>> gate_f, up_f, down_f; // per expert, ggml-dequantized
};

std::vector<float> dequant(ggml_type t, const std::vector<uint8_t>& q, int64_t n) {
    std::vector<float> f((size_t) n);
    ggml_get_type_traits(t)->to_float(q.data(), f.data(), n);
    return f;
}

Weights make_weights(const Case& c, std::mt19937& rng) {
    Weights w;
    const float s_gu = 4.5f / std::sqrt((float) E), s_d = 1.0f / std::sqrt((float) c.nff);
    for (int e = 0; e < NEXP; ++e) {
        w.gate.push_back(quantize(c.gu, normal(rng, (size_t) c.nff * E, s_gu), E, c.nff));
        w.up.push_back(quantize(c.gu, normal(rng, (size_t) c.nff * E, s_gu), E, c.nff));
        w.down.push_back(quantize(c.d, normal(rng, (size_t) E * c.nff, s_d), c.nff, E));
        w.gate_f.push_back(dequant(c.gu, w.gate.back(), (int64_t) c.nff * E));
        w.up_f.push_back(dequant(c.gu, w.up.back(), (int64_t) c.nff * E));
        w.down_f.push_back(dequant(c.d, w.down.back(), (int64_t) E * c.nff));
    }
    return w;
}

struct Routing {
    std::vector<int32_t> ids;   // [T][K]
    std::vector<float> w;       // [T][K]
};
Routing make_routing(std::mt19937& rng) {
    Routing r;
    std::uniform_real_distribution<float> u(0.05f, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> p(NEXP);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng);
        for (int j = 0; j < KUSED; ++j) {
            r.ids.push_back(p[(size_t) j]);
            r.w.push_back(u(rng));
        }
    }
    return r;
}

std::vector<float> exact(const Case& c, const Weights& w, const Routing& r, const std::vector<float>& x, float limit,
                         double* clamped_share) {
    std::vector<float> out((size_t) T * E, 0.f);
    std::vector<double> h((size_t) c.nff);
    int64_t clamped = 0, total = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> acc((size_t) E, 0.0);
        for (int j = 0; j < KUSED; ++j) {
            const int e = r.ids[(size_t) t * KUSED + j];
            const float* xr = x.data() + (size_t) t * E;
            for (int f = 0; f < c.nff; ++f) {
                double g = 0, u = 0;
                const float* gr = w.gate_f[(size_t) e].data() + (size_t) f * E;
                const float* ur = w.up_f[(size_t) e].data() + (size_t) f * E;
                for (int i = 0; i < E; ++i) {
                    g += (double) gr[i] * xr[i];
                    u += (double) ur[i] * xr[i];
                }
                total += 2;
                if (limit > 0) {
                    clamped += (g > limit) + (std::fabs(u) > limit);
                    g = std::min(g, (double) limit);
                    u = std::min(std::max(u, -(double) limit), (double) limit);
                }
                h[(size_t) f] = g / (1.0 + std::exp(-g)) * u;
            }
            const double wt = r.w[(size_t) t * KUSED + j];
            for (int o = 0; o < E; ++o) {
                const float* dr = w.down_f[(size_t) e].data() + (size_t) o * c.nff;
                double s = 0;
                for (int f = 0; f < c.nff; ++f) s += (double) dr[f] * h[(size_t) f];
                acc[(size_t) o] += wt * s;
            }
        }
        for (int o = 0; o < E; ++o) out[(size_t) t * E + o] = (float) acc[(size_t) o];
    }
    if (clamped_share) *clamped_share = (double) clamped / (double) total;
    return out;
}

// build_moe_ffn's expert half on ggml-cpu; `f32`: the same graph over the dequantized weights as F32 tensors (ggml-cpu
// then multiplies in f32 without quantizing the activations: the graph's semantics alone, against the exact result)
std::vector<float> oracle(const Case& c, const Weights& w, const Routing& r, const std::vector<float>& x,
                          bool f32 = false) {
    Oracle o((size_t) 3 << 29);
    auto stack = [&](ggml_type t, const std::vector<std::vector<uint8_t>>& m, const std::vector<std::vector<float>>& mf,
                     int64_t n_in, int64_t rows) {
        if (f32) t = GGML_TYPE_F32;
        ggml_tensor* a = ggml_new_tensor_3d(o.ctx, t, n_in, rows, NEXP);
        const size_t per = ggml_row_size(t, n_in) * (size_t) rows;
        for (int e = 0; e < NEXP; ++e)
            std::memcpy((uint8_t*) a->data + per * e, f32 ? (const void*) mf[(size_t) e].data() : m[(size_t) e].data(),
                        per);
        return a;
    };
    ggml_tensor* gate_exps = stack(c.gu, w.gate, w.gate_f, E, c.nff);
    ggml_tensor* up_exps = stack(c.gu, w.up, w.up_f, E, c.nff);
    ggml_tensor* down_exps = stack(c.d, w.down, w.down_f, c.nff, E);
    ggml_tensor* X = ggml_reshape_3d(o.ctx, o.f32(x, E, T), E, 1, T);
    ggml_tensor* ids = ggml_new_tensor_2d(o.ctx, GGML_TYPE_I32, KUSED, T);
    std::memcpy(ids->data, r.ids.data(), r.ids.size() * 4);
    ggml_tensor* weights = o.f32(r.w, 1, KUSED, T);
    ggml_tensor* up = ggml_mul_mat_id(o.ctx, up_exps, X, ids);       // [nff, K, T]
    ggml_tensor* cur = ggml_mul_mat_id(o.ctx, gate_exps, X, ids);
    cur = ggml_swiglu_clamp(o.ctx, cur, up, LIMIT);
    ggml_tensor* experts = ggml_mul_mat_id(o.ctx, down_exps, cur, ids);   // [E, K, T]
    experts = ggml_mul(o.ctx, experts, weights);
    ggml_tensor* moe = ggml_view_2d(o.ctx, experts, E, T, experts->nb[2], 0);
    for (int j = 1; j < KUSED; ++j)
        moe = ggml_add(o.ctx, moe, ggml_view_2d(o.ctx, experts, E, T, experts->nb[2], (size_t) j * experts->nb[1]));
    moe = ggml_cont(o.ctx, moe);
    o.compute(moe);
    return Oracle::read(moe);
}

std::vector<float> gpu(const Case& c, const Weights& w, const Routing& r, const std::vector<float>& x, float limit,
                       cudaStream_t st) {
    const k::NativeExpertLayout L = k::native_expert_layout(c.gu, c.d, E, c.nff);
    std::vector<uint8_t> blobs((size_t) NEXP * L.bytes);
    for (int e = 0; e < NEXP; ++e) {
        uint8_t* b = blobs.data() + (size_t) e * L.bytes;
        std::memcpy(b, w.gate[(size_t) e].data(), w.gate[(size_t) e].size());
        std::memcpy(b + L.up_off, w.up[(size_t) e].data(), w.up[(size_t) e].size());
        std::memcpy(b + L.down_off, w.down[(size_t) e].data(), w.down[(size_t) e].size());
    }
    // the plan, as GlmPrefill::moe makes it: entries grouped by expert, tokens ascending
    std::vector<int32_t> start(NEXP + 1, 0), rows;
    std::vector<float> wts;
    for (int e = 0; e < NEXP; ++e) {
        start[(size_t) e] = (int32_t) rows.size();
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < KUSED; ++j)
                if (r.ids[(size_t) t * KUSED + j] == e) {
                    rows.push_back(t);
                    wts.push_back(r.w[(size_t) t * KUSED + j]);
                }
    }
    start[NEXP] = (int32_t) rows.size();
    Dev<uint8_t> d_blobs(blobs);
    Dev<float> d_x(x), d_out((size_t) T * E), d_w(wts);
    Dev<int32_t> d_rows(rows);
    Dev<uint16_t> w16((size_t) 2 * c.nff * E), x16((size_t) T * E), h16((size_t) T * c.nff);
    Dev<float> gu((size_t) T * 2 * c.nff), dd((size_t) T * E);
    d_out.zero();
    pg::Gemm16 gemm;
    std::string err;
    if (!gemm.init(st, err)) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(2); }
    const pg::ExpertScratch s{w16.p, x16.p, gu.p, h16.p, dd.p};
    for (int e = 0; e < NEXP; ++e)
        pg::expert_rows(gemm, L, d_blobs.p + (size_t) e * L.bytes, d_x.p, E, d_rows.p + start[(size_t) e],
                        d_w.p + start[(size_t) e], start[(size_t) e + 1] - start[(size_t) e], limit, s, d_out.p, st);
    GLM_CHECK(cudaStreamSynchronize(st));
    return d_out.down();
}

bool run_case(const Case& c, unsigned seed, cudaStream_t st, bool control) {
    std::mt19937 rng(seed);
    std::printf("[moe] %s, n_ff %d, %d experts, top-%d, %d tokens, seed %u\n", c.name, c.nff, NEXP, KUSED, T, seed);
    if (!k::native_expert_supported(c.gu, c.d, E, c.nff)) {   // GlmPrefill's per-layer format check
        std::printf("  native_expert_supported: false  FAIL\n");
        return false;
    }
    const Weights w = make_weights(c, rng);
    const Routing r = make_routing(rng);
    const std::vector<float> x = normal(rng, (size_t) T * E, 1.0f);
    double share = 0;
    const auto ex = exact(c, w, r, x, LIMIT, &share);
    const auto orc = oracle(c, w, r, x);
    const auto g = gpu(c, w, r, x, LIMIT, st);
    std::printf("  %.2f%% of the gate/up pre-activations are past the clamp limit %.0f\n", 100.0 * share, LIMIT);
    bool ok = true;
    ok &= report("GPU vs exact", compare(g.data(), ex.data(), g.size()), TOL_EXACT);
    const auto orc32 = oracle(c, w, r, x, true);
    ok &= report("oracle graph on F32 weights vs exact", compare(orc32.data(), ex.data(), orc32.size()), 1e-5);
    const Err oe = compare(orc.data(), ex.data(), orc.size());
    std::printf("  %-34s rel %.3e  rms_rel %.3e  (the oracle's own Q8_K rounding)\n", "oracle vs exact", oe.rel(),
                oe.rms_rel());
    // relative to max |exact| on both sides of the triangle inequality
    Err go = compare(g.data(), orc.data(), g.size());
    go.max_ref = oe.max_ref;
    ok &= report("GPU vs oracle (ggml-cpu f1cee99)", go, oe.rel() + TOL_EXACT);
    if (control) {
        const auto off = gpu(c, w, r, x, 0.0f, st);
        const Err e = compare(off.data(), ex.data(), off.size());
        const bool caught = e.rel() > TOL_EXACT;
        std::printf("  %-34s rel %.3e against exact: %s\n", "control: clamp off", e.rel(),
                    caught ? "detected (PASS)" : "NOT detected (FAIL)");
        ok &= caught;
    }
    return ok;
}

void timing(cudaStream_t st) {
    constexpr int NFF = 2048, ROWS = 512;
    const k::NativeExpertLayout L = k::native_expert_layout(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, E, NFF);
    std::mt19937 rng(9);
    Dev<uint8_t> blob(L.bytes);
    GLM_CHECK(cudaMemset(blob.p, 0x11, L.bytes));
    std::vector<int32_t> rows(ROWS);
    std::iota(rows.begin(), rows.end(), 0);
    Dev<int32_t> d_rows(rows);
    Dev<float> x(normal(rng, (size_t) ROWS * E, 1.0f)), w(std::vector<float>(ROWS, 0.5f)), out((size_t) ROWS * E);
    Dev<uint16_t> w16((size_t) 2 * NFF * E), x16((size_t) ROWS * E), h16((size_t) ROWS * NFF);
    Dev<float> gu((size_t) ROWS * 2 * NFF), dd((size_t) ROWS * E);
    pg::Gemm16 gemm;
    std::string err;
    if (!gemm.init(st, err)) return;
    const pg::ExpertScratch s{w16.p, x16.p, gu.p, h16.p, dd.p};
    const double us = median_us([&] { pg::expert_rows(gemm, L, blob.p, x.p, E, d_rows.p, w.p, ROWS, LIMIT, s, out.p, st); },
                                st, 20, 3, 3);
    std::printf("[timing] one IQ3_S/IQ4_XS expert (n_ff 2048) over %d rows: %.1f us (%.1f TFLOPS)\n", ROWS, us,
                2.0 * 3 * NFF * (double) E * ROWS / us / 1e6);
}

}  // namespace

int main() {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    const Case cases[] = {{GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, 512, "IQ3_S / IQ4_XS"},
                          {GGML_TYPE_IQ3_S, GGML_TYPE_Q6_K, 512, "IQ3_S / Q6_K"},
                          {GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K, 512, "IQ4_XS / Q6_K"},
                          {GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, 2048, "IQ3_S / IQ4_XS"},
                          {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, 2048, "IQ2_S / IQ3_S (UD-IQ3_XXS)"},
                          {GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, 2048, "IQ2_S / IQ4_XS (UD-IQ3_XXS)"}};
    bool ok = true;
    unsigned seed = 1;
    for (const Case& c : cases) {
        ok &= run_case(c, seed, st, seed == 1);
        ++seed;
    }
    timing(st);
    std::printf("prefill_moe_parity: %s\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
