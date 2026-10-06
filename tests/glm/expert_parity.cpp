// tests/glm/expert_parity.cpp - GLM-5.3-Flash's routed-expert changes: the clamped SwiGLU epilogue
// (NativeExpertLayout::swiglu_limit on the GPU, native_gu_rows_clamp on the CPU pool), Q6_K and IQ3_S down rows
// (native_expert_grouped, iq_dequant_f32; the CPU pool through ggml-cpu's vec_dot) and the Q6_K embedding rows.
//
// Formats and shapes of the GGUFs, all 4096 x 2048: UD-IQ4_XS's IQ3_S/IQ4_XS (39 layers), IQ3_S/Q6_K (layers 12, 44),
// IQ4_XS/Q6_K (layer 11); UD-IQ3_XXS's IQ2_S/IQ3_S (39 layers: IQ3_S as a down projection), IQ2_S/IQ4_XS (layers 12,
// 44) and IQ3_S/IQ4_XS (layer 11, the first case); plus Qwen's IQ3_S/IQ4_NL at 2560 x 640 with the clamp on.
// Random weights quantized by ggml's quantizer (Strata's pinned ggml, whose Q6_K / IQ2_S / IQ3_S / IQ4_XS
// quantize/dequantize/vec_dot are those of f1cee99: ggml-quants.c differs between the two only in quantize_iq1_s,
// ggml-cpu's x86 quants not at all), scaled so that ~30% of the gate/up values pass +-10.
// Six experts, four tokens, 1..4 tokens per expert.
//
// Oracles and tolerances (of max |ref|):
//   GPU gate/up        double sum of ggml's dequantized weights x the q8_1 activation the GPU used     1e-5
//                      (IQ2_S: llama.cpp CUDA's own integer-truncating formula, emulated; see iq2s_cuda_dot)
//   GPU h (V1 kernels) swiglu_clamped(g, u, 10) on the GPU's own g, u (std::exp vs __expf)          1e-6
//   GPU down           double sum of ggml's dequantized down rows x the GPU's own q8_1 h            1e-5
//   GPU dequant        ggml's dequantize_row_q6_K / _iq3_s (the prompt path's Q6_K, IQ3_S down rows)  bitwise
//   GPU embedding rows ggml's dequantize_row_q6_K (iq_embed_rows: UD-IQ3_XXS's Q6_K token_embd)      bitwise
//   CPU ff, 1 token    ggml's vec_dot (the path's own dots) + swiglu_clamped                          bitwise
//   CPU ff, 2-4 tokens the same reference (the multi-token AVX kernels round on their own, #152)     1e-6
//   CPU down rows      double sum of ggml's dequantized rows x the dequantized Q8_K h                1e-5
// and BITWISE: swiglu_limit 1e30 (the clamp branch taken, never binding) equals swiglu_limit 0 on every path, the
// default GPU kernels equal the V1 ones with the clamp on, and the clamp changes the output (it is active).
// (Qwen's own bit-identity against the base tree is tests/glm/expert_dump.cpp's.)
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/native_expert_clamp.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include "glm_test_util.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace glmtest;
namespace k = strata::kernels;

namespace {

constexpr int E = 6, T = 4;
const std::vector<std::vector<int>> kToks = {{0, 1, 2, 3}, {1}, {0, 2}, {3}, {0, 1, 2}, {2}};

std::vector<uint8_t> quant(int type, const std::vector<float>& w, int64_t n_in, int64_t rows) {
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, n_in) * (size_t) rows);
    ggml_quantize_init((ggml_type) type);
    ggml_quantize_chunk((ggml_type) type, w.data(), q.data(), 0, rows, n_in, nullptr);
    return q;
}
std::vector<float> dequant(int type, const uint8_t* q, int64_t n) {
    std::vector<float> f((size_t) n);
    if (type == GGML_TYPE_Q8_K) {   // block_q8_K {float d; int8_t qs[256]; int16_t bsums[16]}: ggml has no to_float
        for (int64_t b = 0; b < n / 256; ++b) {
            const uint8_t* blk = q + b * 292;
            float d;
            std::memcpy(&d, blk, 4);
            for (int i = 0; i < 256; ++i) f[(size_t) (b * 256 + i)] = d * (float) (int8_t) blk[4 + i];
        }
        return f;
    }
    ggml_get_type_traits((ggml_type) type)->to_float(q, f.data(), n);
    return f;
}
// q8_1 blocks (half2 {d, sum}, 32 x int8) to floats d * q
std::vector<float> dequant_q8_1(const uint8_t* q, int64_t n) {
    std::vector<float> f((size_t) n);
    for (int64_t b = 0; b < n / 32; ++b) {
        const uint8_t* blk = q + b * 36;
        ggml_fp16_t dh;
        std::memcpy(&dh, blk, 2);
        const float d = ggml_fp16_to_fp32(dh);
        for (int i = 0; i < 32; ++i) f[(size_t) (b * 32 + i)] = d * (float) (int8_t) blk[4 + i];
    }
    return f;
}

// llama.cpp CUDA's vec_dot_iq2_s_q8_1 (ggml-cuda/vecdotq.cuh 1124-1167 at f1cee99; iq_kernels.cu transcribes it) applies
// IQ2_S's scale (ls + 1/2) / 4 to the integer sums of each 32 values with two truncating divisions:
//   sumi = (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4,   result = (d_w * d_x) * sumi
// (s0, s1 the dp4a sums of values 0-15 and 16-31).  ggml-cpu and the f64 reference use the exact scale, so the GPU's
// IQ2_S gate/up rows differ from them by up to 1.125 |d_w d_x| per 32 values (~3e-5 of max |ref| here).  This is the
// same formula on the host: the signed grid values come back exactly from ggml's dequantized weights (y / db, small
// integers), the activations are the GPU's own q8_1 codes.  Per output row: the emulated value (the float
// products summed in double), the truncation bound sum 1.125 |d_w d_x| over its blocks, and sum |term| (the scale of
// the GPU's float rounding: 128 terms summed in float, allowed 1e-5 of it).
struct Iq2sRow { double v, bound, mag; };
Iq2sRow iq2s_cuda_dot(const uint8_t* blocks, const float* wdq, const uint8_t* xq, int64_t n) {
    Iq2sRow r{0, 0, 0};
    for (int64_t b = 0; b < n / 256; ++b) {
        const uint8_t* blk = blocks + b * 82;   // block_iq2_s: d (fp16), qs[64], qh[8], scales[8]
        ggml_fp16_t dh;
        std::memcpy(&dh, blk, 2);
        const float d = ggml_fp16_to_fp32(dh);
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* xb = xq + (b * 8 + ib) * 36;   // q8_1: half2 {d, sum}, 32 x int8
            ggml_fp16_t xh;
            std::memcpy(&xh, xb, 2);
            const float dx = ggml_fp16_to_fp32(xh);
            const int ls[2] = {blk[74 + ib] & 0x0F, blk[74 + ib] >> 4};
            int sh[2] = {0, 0};
            for (int h = 0; h < 2; ++h) {
                const float db = d * (0.5f + (float) ls[h]) * 0.25f;   // dequantize_row_iq2_s's scale
                for (int j = 0; j < 16; ++j) {
                    const int64_t k = b * 256 + ib * 32 + h * 16 + j;
                    const int g = db != 0.f ? (int) std::lround(wdq[k] / db) : 0;
                    sh[h] += g * (int) (int8_t) xb[4 + h * 16 + j];
                }
            }
            const int sumi = (sh[0] * ls[0] + sh[1] * ls[1] + (sh[0] + sh[1]) / 2) / 4;
            const float dd = d * dx;
            r.v += (double) (dd * (float) sumi);
            r.mag += std::fabs((double) (dd * (float) sumi));
            r.bound += 1.125 * std::fabs((double) dd);
        }
    }
    return r;
}

struct Expert {
    int gu, dn;
    int64_t H, FF;
    k::NativeExpertLayout L;
    std::vector<uint8_t> blobs;
    std::vector<float> x;
};

Expert make(int gu, int dn, int64_t H, int64_t FF, unsigned seed) {
    std::mt19937 rng(seed);
    Expert X{gu, dn, H, FF, k::native_expert_layout(gu, dn, H, FF), {}, {}};
    const float sd = 9.6f / std::sqrt((float) H);   // gate/up ~ N(0, 9.6): ~30% beyond +-10
    for (int e = 0; e < E; ++e) {
        const auto g = quant(gu, normal(rng, (size_t) (H * FF), sd), H, FF);
        const auto u = quant(gu, normal(rng, (size_t) (H * FF), sd), H, FF);
        const auto d = quant(dn, normal(rng, (size_t) (H * FF), 1.0f / std::sqrt((float) FF)), FF, H);
        X.blobs.insert(X.blobs.end(), g.begin(), g.end());
        X.blobs.insert(X.blobs.end(), u.begin(), u.end());
        X.blobs.insert(X.blobs.end(), d.begin(), d.end());
    }
    X.x = normal(rng, (size_t) (T * H), 1.0f);
    return X;
}

struct GpuRun {
    std::vector<float> out, gate, up, h;
    std::vector<uint8_t> xq, hq;
};

GpuRun gpu(const Expert& X, float limit, bool v1, cudaStream_t st) {
    std::vector<int32_t> start = {0}, ent_tok, ent_dst;
    for (int e = 0; e < E; ++e) {
        for (int t : kToks[(size_t) e]) { ent_tok.push_back(t); ent_dst.push_back((int32_t) ent_dst.size()); }
        start.push_back((int32_t) ent_tok.size());
    }
    const int n = (int) ent_tok.size();
    Dev<uint8_t> blobs(X.blobs), xq((size_t) T * (X.H / 32) * 36), scr(k::native_expert_scratch_bytes(n, X.FF));
    Dev<float> x(X.x), out((size_t) n * X.H);
    std::vector<unsigned long long> ptr;
    for (int e = 0; e < E; ++e) ptr.push_back((unsigned long long) (blobs.p + (size_t) e * X.L.bytes));
    Dev<unsigned long long> dptr(ptr);
    Dev<int32_t> dstart(start), dng(std::vector<int32_t>{E}), dtok(ent_tok), ddst(ent_dst);
    k::NativeExpertLayout L = X.L;
    L.swiglu_limit = limit;
    k::quantize_q8_1_rows(x.p, T, X.H, xq.p, st);
    k::native_grouped_set_v1(v1);
    out.zero();
    k::native_expert_grouped(L, dptr.p, dstart.p, dng.p, ddst.p, dtok.p, E, n, xq.p, scr.p, out.p, st);
    k::native_grouped_set_v1(false);
    GLM_CHECK(cudaStreamSynchronize(st));
    GpuRun r;
    r.out = out.down();
    r.xq = xq.down();
    const auto s = scr.down();
    const size_t f = (size_t) n * (size_t) X.FF * 4, fa = (f + 255) & ~(size_t) 255;
    r.gate.assign((const float*) s.data(), (const float*) s.data() + (size_t) n * X.FF);
    r.up.assign((const float*) (s.data() + fa), (const float*) (s.data() + fa) + (size_t) n * X.FF);
    r.h.assign((const float*) (s.data() + 2 * fa), (const float*) (s.data() + 2 * fa) + (size_t) n * X.FF);
    r.hq.assign(s.data() + 3 * fa, s.data() + 3 * fa + (size_t) n * (X.FF / 32) * 36);
    return r;
}

bool gpu_checks(const Expert& X, cudaStream_t st) {
    bool ok = true;
    const GpuRun off = gpu(X, 0.0f, false, st), huge = gpu(X, 1e30f, false, st);
    const GpuRun on = gpu(X, 10.0f, false, st), on_v1 = gpu(X, 10.0f, true, st);
    const GpuRun off_v1 = gpu(X, 0.0f, true, st), huge_v1 = gpu(X, 1e30f, true, st);
    ok &= bitwise("GPU limit 1e30 == limit 0", huge.out.data(), off.out.data(), off.out.size());
    ok &= bitwise("GPU V1: limit 1e30 == limit 0", huge_v1.out.data(), off_v1.out.data(), off.out.size());
    ok &= bitwise("GPU clamp: default == V1 kernels", on.out.data(), on_v1.out.data(), on.out.size());
    // the clamp is active
    size_t clamped = 0;
    for (size_t i = 0; i < on_v1.gate.size(); ++i) clamped += on_v1.gate[i] > 10.f || std::fabs(on_v1.up[i]) > 10.f;
    const Err moved = compare(on.out.data(), off.out.data(), on.out.size());
    const bool active = clamped > on_v1.gate.size() / 10 && moved.max_abs > 0;
    std::printf("  %-34s %zu of %zu gate/up pairs beyond 10; output moved by up to %.3e  %s\n", "GPU clamp active",
                clamped, on_v1.gate.size(), moved.max_abs, active ? "PASS" : "FAIL");
    ok &= active;
    // h against the host formula on the GPU's own gate/up
    std::vector<float> h_ref(on_v1.h.size());
    for (size_t i = 0; i < h_ref.size(); ++i) h_ref[i] = k::cpu::swiglu_clamped(on_v1.gate[i], on_v1.up[i], 10.0f);
    ok &= report("GPU h = silu(min(g,10))*clamp(u)", compare(on_v1.h.data(), h_ref.data(), h_ref.size()), 1e-6);
    // gate/up and down against double references
    const auto xd = dequant_q8_1(on.xq.data(), (int64_t) T * X.H);
    const auto hd = dequant_q8_1(on.hq.data(), (int64_t) on.out.size() / X.H * X.FF);
    std::vector<float> g_ref(on.gate.size()), u_ref(on.up.size()), o_ref(on.out.size());
    const bool iq2s = X.gu == 22;   // the llama.cpp CUDA formula above, and its distance to the exact one
    std::vector<float> g_emu(iq2s ? on.gate.size() : 0), u_emu(iq2s ? on.up.size() : 0);
    double trunc_excess = -1e30;    // max over rows of |gpu - exact| - (bound + 1e-5 sum|term|), must stay <= 0
    int ent = 0;
    for (int e = 0; e < E; ++e) {
        const uint8_t* blob = X.blobs.data() + (size_t) e * X.L.bytes;
        const auto wg = dequant(X.gu, blob, X.H * X.FF), wu = dequant(X.gu, blob + X.L.up_off, X.H * X.FF);
        const auto wd = dequant(X.dn, blob + X.L.down_off, X.H * X.FF);
        for (int t : kToks[(size_t) e]) {
            const float* xt = xd.data() + (size_t) t * X.H;
            for (int64_t r = 0; r < X.FF; ++r) {
                double sg = 0, su = 0;
                for (int64_t c = 0; c < X.H; ++c) {
                    sg += (double) wg[(size_t) (r * X.H + c)] * xt[c];
                    su += (double) wu[(size_t) (r * X.H + c)] * xt[c];
                }
                g_ref[(size_t) (ent * X.FF + r)] = (float) sg;
                u_ref[(size_t) (ent * X.FF + r)] = (float) su;
                if (iq2s) {
                    const uint8_t* xq = on.xq.data() + (size_t) t * (X.H / 32) * 36;
                    const Iq2sRow eg = iq2s_cuda_dot(blob + (size_t) r * X.L.gu_row, wg.data() + r * X.H, xq, X.H);
                    const Iq2sRow eu = iq2s_cuda_dot(blob + X.L.up_off + (size_t) r * X.L.gu_row, wu.data() + r * X.H,
                                                     xq, X.H);
                    const size_t i = (size_t) (ent * X.FF + r);
                    g_emu[i] = (float) eg.v;
                    u_emu[i] = (float) eu.v;
                    trunc_excess = std::max({trunc_excess, std::fabs(on.gate[i] - sg) - eg.bound - 1e-5 * eg.mag,
                                             std::fabs(on.up[i] - su) - eu.bound - 1e-5 * eu.mag});
                }
            }
            const float* ht = hd.data() + (size_t) ent * X.FF;
            for (int64_t r = 0; r < X.H; ++r) {
                double s = 0;
                for (int64_t c = 0; c < X.FF; ++c) s += (double) wd[(size_t) (r * X.FF + c)] * ht[c];
                o_ref[(size_t) (ent * X.H + r)] = (float) s;
            }
            ++ent;
        }
    }
    if (iq2s) {
        ok &= report("GPU gate (vs llama.cpp CUDA's IQ2_S)", compare(on.gate.data(), g_emu.data(), g_emu.size()), 1e-5);
        ok &= report("GPU up (vs llama.cpp CUDA's IQ2_S)", compare(on.up.data(), u_emu.data(), u_emu.size()), 1e-5);
        const Err eg = compare(on.gate.data(), g_ref.data(), g_ref.size());
        const Err eu = compare(on.up.data(), u_ref.data(), u_ref.size());
        std::printf("  %-34s rel %.3e / %.3e (gate / up; llama.cpp CUDA's integer truncation)\n",
                    "  the same vs the exact f64 dot", eg.rel(), eu.rel());
        const bool within = trunc_excess <= 0.0;
        std::printf("  %-34s every row within its truncation bound (worst excess %.3e)  %s\n",
                    "  ... and its bound", trunc_excess, within ? "PASS" : "FAIL");
        ok &= within;
    } else {
        ok &= report("GPU gate (vs f64 dot)", compare(on.gate.data(), g_ref.data(), g_ref.size()), 1e-5);
        ok &= report("GPU up (vs f64 dot)", compare(on.up.data(), u_ref.data(), u_ref.size()), 1e-5);
    }
    ok &= report("GPU down (vs f64 dot on its q8_1 h)", compare(on.out.data(), o_ref.data(), o_ref.size()), 1e-5);
    if (X.dn == 14 || X.dn == 21) {   // the prompt path's dequantizer for the Q6_K / IQ3_S down rows
        const uint8_t* down0 = X.blobs.data() + X.L.down_off;
        const auto ref = dequant(X.dn, down0, X.H * X.FF);
        Dev<uint8_t> dw(std::vector<uint8_t>(down0, down0 + (size_t) X.H * X.L.d_row));
        Dev<float> out((size_t) (X.H * X.FF));
        k::iq_dequant_f32(X.dn, dw.p, X.H * X.FF, out.p, st);
        GLM_CHECK(cudaStreamSynchronize(st));
        ok &= bitwise(X.dn == 14 ? "GPU Q6_K dequant == ggml's" : "GPU IQ3_S dequant == ggml's", out.down().data(),
                      ref.data(), ref.size());
        if (X.dn == 14) {
            // UD-IQ3_XXS's Q6_K token_embd through iq_embed_rows (the down matrix as a table of H rows of FF values),
            // and the gate: Q6_K only for glm5-next (glm_embed_type_supported), a Qwen file as before
            const std::vector<int32_t> ids = {0, 7, (int32_t) X.H - 1, 1234, 7};
            Dev<int32_t> dids(ids);
            Dev<float> rows(ids.size() * (size_t) X.FF);
            k::iq_embed_rows(14, dw.p, X.L.d_row, dids.p, (int64_t) ids.size(), X.FF, rows.p, st);
            GLM_CHECK(cudaStreamSynchronize(st));
            std::vector<float> want;
            for (int32_t id : ids) want.insert(want.end(), ref.begin() + (size_t) id * X.FF, ref.begin() + (size_t) (id + 1) * X.FF);
            ok &= bitwise("GPU Q6_K embedding rows == ggml's", rows.down().data(), want.data(), want.size());
            const bool gate = k::glm_embed_type_supported(14) && !k::embed_type_supported(14);
            std::printf("  %-34s %s\n", "Q6_K embedding: glm5-next only", gate ? "PASS" : "FAIL");
            ok &= gate;
        }
    }
    return ok;
}

// The CPU pool: native_gu_rows_clamp on ggml's path (1 token) and the multi-token kernels (2-4), and the down rows.
bool cpu_checks(const Expert& X) {
    bool ok = true;
    k::cpu::NativeFmt f;
    std::string err;
    if (!k::cpu::native_fmt(X.gu, X.dn, X.H, X.FF, f, err)) {
        // native_fmt caps the activation sizes at Qwen's (native_expert.hpp, kNativeHBytes; generalized by the
        // geometry worker): fill the same fields here from ggml-cpu's traits to test the arithmetic itself
        std::printf("  (native_fmt refuses %lld x %lld: \"%s\" - the format is filled here instead)\n", (long long) X.H,
                    (long long) X.FF, err.c_str());
        const auto* tg = ggml_get_type_traits_cpu((ggml_type) X.gu);
        const auto* td = ggml_get_type_traits_cpu((ggml_type) X.dn);
        f.gu_type = X.gu;
        f.d_type = X.dn;
        f.gu_act = (int) tg->vec_dot_type;
        f.d_act = (int) td->vec_dot_type;
        f.n_embd = X.H;
        f.n_ff = X.FF;
        f.gu_row = X.L.gu_row;
        f.d_row = X.L.d_row;
        f.up_off = X.L.up_off;
        f.down_off = X.L.down_off;
        f.bytes = X.L.bytes;
        f.act_bytes = ggml_row_size(tg->vec_dot_type, X.H);
        f.h_bytes = ggml_row_size(td->vec_dot_type, X.FF);
    }
    std::vector<std::vector<uint8_t>> act(T, std::vector<uint8_t>(f.act_bytes));
    for (int t = 0; t < T; ++t) k::cpu::native_quant_act(f, X.x.data() + (size_t) t * X.H, act[(size_t) t].data());
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) X.gu);
    Err e_single, e_multi, e_down;
    bool same_off = true, exact_single = true, moved = false;
    for (int e = 0; e < E; ++e) {
        const uint8_t* blob = X.blobs.data() + (size_t) e * X.L.bytes;
        const int nt = (int) kToks[(size_t) e].size();
        std::vector<const void*> a;
        for (int t : kToks[(size_t) e]) a.push_back(act[(size_t) t].data());
        auto rows = [&](float limit) {
            std::vector<std::vector<float>> ff((size_t) nt, std::vector<float>((size_t) X.FF));
            std::vector<float*> p;
            for (auto& v : ff) p.push_back(v.data());
            k::cpu::native_gu_rows_clamp(f, blob, a.data(), nt, p.data(), 0, (int) X.FF, limit);
            return ff;
        };
        const auto ff0 = rows(0.0f), ffh = rows(1e30f), ff10 = rows(10.0f);
        for (int j = 0; j < nt; ++j) {
            same_off &= std::memcmp(ff0[(size_t) j].data(), ffh[(size_t) j].data(), (size_t) X.FF * 4) == 0;
            // the reference: ggml-cpu's own dot for this row and token, then the clamped SwiGLU
            std::vector<float> ref((size_t) X.FF);
            for (int64_t r = 0; r < X.FF; ++r) {
                float g = 0, u = 0;
                tg->vec_dot((int) X.H, &g, 0, blob + (size_t) r * f.gu_row, 0, a[(size_t) j], 0, 1);
                tg->vec_dot((int) X.H, &u, 0, blob + f.up_off + (size_t) r * f.gu_row, 0, a[(size_t) j], 0, 1);
                ref[(size_t) r] = k::cpu::swiglu_clamped(g, u, 10.0f);
            }
            const Err er = compare(ff10[(size_t) j].data(), ref.data(), ref.size());
            if (nt == 1) {
                e_single.merge(er);
                exact_single &= er.max_abs == 0;
            } else {
                e_multi.merge(er);
            }
            moved |= std::memcmp(ff0[(size_t) j].data(), ff10[(size_t) j].data(), (size_t) X.FF * 4) != 0;
        }
        // down rows (Q6_K, IQ3_S, IQ4_XS: ggml-cpu's vec_dot against Q8_K) on the clamped h
        std::vector<std::vector<uint8_t>> hq((size_t) nt, std::vector<uint8_t>(f.h_bytes));
        std::vector<const void*> hp;
        std::vector<std::vector<float>> o((size_t) nt, std::vector<float>((size_t) X.H));
        std::vector<float*> op;
        for (int j = 0; j < nt; ++j) {
            k::cpu::native_quant_h(f, ff10[(size_t) j].data(), hq[(size_t) j].data());
            hp.push_back(hq[(size_t) j].data());
            op.push_back(o[(size_t) j].data());
        }
        k::cpu::native_down_rows(f, blob, hp.data(), nt, op.data(), 0, (int) X.H);
        const auto wd = dequant(X.dn, blob + f.down_off, X.H * X.FF);
        for (int j = 0; j < nt; ++j) {
            const auto hdq = dequant(f.d_act, hq[(size_t) j].data(), X.FF);
            std::vector<float> ref((size_t) X.H);
            for (int64_t r = 0; r < X.H; ++r) {
                double s = 0;
                for (int64_t c = 0; c < X.FF; ++c) s += (double) wd[(size_t) (r * X.FF + c)] * hdq[(size_t) c];
                ref[(size_t) r] = (float) s;
            }
            e_down.merge(compare(o[(size_t) j].data(), ref.data(), ref.size()));
        }
    }
    std::printf("  %-34s %s\n", "CPU limit 1e30 == limit 0 (bitwise)", same_off ? "PASS" : "FAIL");
    std::printf("  %-34s %s (max_abs %.3e)\n", "CPU 1 token == ggml dot + clamp", exact_single ? "bitwise PASS" : "FAIL",
                e_single.max_abs);
    std::printf("  %-34s %s\n", "CPU clamp active", moved ? "PASS" : "FAIL");
    ok &= same_off && exact_single && moved;
    ok &= report("CPU 2-4 tokens vs ggml dot + clamp", e_multi, 1e-6);
    ok &= report("CPU down rows (vs f64 dot)", e_down, 1e-5);
    return ok;
}

}  // namespace

int main() {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    ggml_cpu_init();
    struct Cfg { int gu, dn; int64_t H, FF; const char* what; };
    const Cfg cfgs[] = {{21, 23, 4096, 2048, "IQ3_S / IQ4_XS (UD-IQ4_XS 39 layers; UD-IQ3_XXS layer 11)"},
                        {21, 14, 4096, 2048, "IQ3_S / Q6_K (UD-IQ4_XS layers 12, 44)"},
                        {23, 14, 4096, 2048, "IQ4_XS / Q6_K (UD-IQ4_XS layer 11)"},
                        {22, 21, 4096, 2048, "IQ2_S / IQ3_S (UD-IQ3_XXS, 39 layers)"},
                        {22, 23, 4096, 2048, "IQ2_S / IQ4_XS (UD-IQ3_XXS layers 12, 44)"},
                        {21, 20, 2560, 640, "IQ3_S / IQ4_NL (Qwen's format, clamp on)"}};
    bool ok = true;
    unsigned seed = 100;
    for (const Cfg& c : cfgs) {
        std::printf("[experts] %s, %lld x %lld\n", c.what, (long long) c.H, (long long) c.FF);
        if (!k::native_expert_supported(c.gu, c.dn, c.H, c.FF)) {
            std::printf("  native_expert_supported: false  FAIL\n");
            ok = false;
            continue;
        }
        const Expert X = make(c.gu, c.dn, c.H, c.FF, seed++);
        ok &= gpu_checks(X, st);
        ok &= cpu_checks(X);
        {
            // one token through one expert (the decode case): the GPU grouped call, clamp on
            std::vector<int32_t> start = {0, 1}, tok = {0}, dst = {0};
            Dev<uint8_t> blobs(std::vector<uint8_t>(X.blobs.begin(), X.blobs.begin() + (long) X.L.bytes));
            Dev<uint8_t> xq((size_t) (X.H / 32) * 36), scr(k::native_expert_scratch_bytes(8, X.FF));
            Dev<float> x(X.x), out((size_t) X.H * 8);
            Dev<unsigned long long> ptr(std::vector<unsigned long long>{(unsigned long long) blobs.p});
            Dev<int32_t> dstart(start), dng(std::vector<int32_t>{1}), dtok(tok), ddst(dst);
            k::NativeExpertLayout L = X.L;
            L.swiglu_limit = 10.0f;
            k::quantize_q8_1_rows(x.p, 1, X.H, xq.p, st);
            std::printf("  [timing] one expert, one token, clamp on: %.2f us (median of 100, best of 5; floor %.2f us)\n",
                        median_us([&] { k::native_expert_grouped(L, ptr.p, dstart.p, dng.p, ddst.p, dtok.p, 1, 1, xq.p,
                                                                 scr.p, out.p, st); }, st), floor_us(st));
        }
    }
    std::printf("expert_parity: %s\n", ok ? "PASS" : "FAIL");
    cudaStreamDestroy(st);
    return ok ? 0 : 1;
}
