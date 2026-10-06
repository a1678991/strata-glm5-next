// tests/glm/pool_parity.cpp - GLM-5.3-Flash's RAM-tier experts on the CPU pool (src/core/glm_pool.cpp): the pool's
// clamped native phases and the combine that reads the CPU's rows.
//
//   1. ExpertPool::run_split_multi_native(f, jobs, n, 10) - the rows split over 4 workers + the host - equals the same
//      row functions called on one thread over all rows (native_quant_act, native_gu_rows_clamp(..., 10),
//      native_quant_h, native_down_rows), BITWISE, for windows of 1, 3 and 8 tokens routed 8 ways over 6 experts
//      (1..8 tokens per expert), in GLM's three expert formats at 4096 x 2048.  Rows are independent dot products, so
//      how the pool cuts them cannot change a bit.
//   2. swiglu_limit 0 through the pool equals native_gu_rows (the Qwen call) bitwise, and 10 changes the output (the
//      clamp is active on these weights).
//   3. glm_moe_combine with cpu_rows / cpu_map equals the GPU-only combine bitwise when the CPU rows hold the same
//      floats as the parts rows they replace (T = 1..8, k = 8, random maps), and differs when they do not.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/native_expert_clamp.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/glm_moe.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include "glm_test_util.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace glmtest;
namespace kc = strata::kernels::cpu;
namespace k = strata::kernels;

namespace {

constexpr int NE = 6, K = 8;

std::vector<uint8_t> quant(int type, const std::vector<float>& w, int64_t n_in, int64_t rows) {
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, n_in) * (size_t) rows);
    ggml_quantize_init((ggml_type) type);
    ggml_quantize_chunk((ggml_type) type, w.data(), q.data(), 0, rows, n_in, nullptr);
    return q;
}

// out[e][t] rows of n_embd for window `route` (route[t][j] = expert): through the pool, or on one thread directly
struct Window {
    int T;
    std::vector<int> route;   // [T][K]
};

std::vector<float> run(const kc::NativeFmt& f, const std::vector<uint8_t>& blobs, const std::vector<float>& x,
                       const Window& w, float limit, kc::ExpertPool* pool) {
    std::vector<uint8_t> act((size_t) w.T * kc::kNativeActBytes);
    for (int t = 0; t < w.T; ++t) kc::native_quant_act(f, x.data() + (size_t) t * f.n_embd, act.data() + (size_t) t * kc::kNativeActBytes);
    std::vector<float> out((size_t) w.T * K * f.n_embd, -1.0f);
    std::vector<kc::ExpertJobMulti> jobs;
    for (int e = 0; e < NE; ++e) {
        kc::ExpertJobMulti j;
        j.blob = blobs.data() + (size_t) e * f.bytes;
        for (int i = 0; i < w.T * K; ++i)
            if (w.route[(size_t) i] == e) {
                j.nact[j.nt] = act.data() + (size_t) (i / K) * kc::kNativeActBytes;
                j.out[j.nt] = out.data() + (size_t) i * f.n_embd;
                ++j.nt;
            }
        if (j.nt > 0) jobs.push_back(j);
    }
    if (pool != nullptr) {
        pool->run_split_multi_native(f, jobs.data(), (int) jobs.size(), limit);
        return out;
    }
    std::vector<float> ffb((size_t) kc::MAXT * f.n_ff);
    std::vector<uint8_t> hq((size_t) kc::MAXT * kc::kNativeHBytes);
    for (const kc::ExpertJobMulti& j : jobs) {
        float* ff[kc::MAXT];
        const void* h[kc::MAXT];
        for (int t = 0; t < j.nt; ++t) { ff[t] = ffb.data() + (size_t) t * f.n_ff; h[t] = hq.data() + (size_t) t * kc::kNativeHBytes; }
        if (limit > 0) kc::native_gu_rows_clamp(f, j.blob, j.nact, j.nt, ff, 0, (int) f.n_ff, limit);
        else kc::native_gu_rows(f, j.blob, j.nact, j.nt, ff, 0, (int) f.n_ff);
        for (int t = 0; t < j.nt; ++t) kc::native_quant_h(f, ff[t], hq.data() + (size_t) t * kc::kNativeHBytes);
        kc::native_down_rows(f, j.blob, h, j.nt, j.out, 0, (int) f.n_embd);
    }
    return out;
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool pool_checks(kc::ExpertPool& pool) {
    struct Cfg { int gu, dn; const char* what; };
    const Cfg cfgs[] = {{21, 23, "IQ3_S / IQ4_XS"}, {21, 14, "IQ3_S / Q6_K"}, {23, 14, "IQ4_XS / Q6_K"}};
    const int64_t H = 4096, FF = 2048;
    bool ok = true;
    unsigned seed = 300;
    for (const Cfg& c : cfgs) {
        kc::NativeFmt f;
        std::string err;
        if (!kc::native_fmt(c.gu, c.dn, H, FF, f, err)) { std::printf("  %s: %s  FAIL\n", c.what, err.c_str()); ok = false; continue; }
        std::mt19937 rng(seed++);
        const float sd = 9.6f / std::sqrt((float) H);   // ~30% of gate/up beyond +-10, as expert_parity
        std::vector<uint8_t> blobs;
        for (int e = 0; e < NE; ++e) {
            const auto g = quant(c.gu, normal(rng, (size_t) (H * FF), sd), H, FF);
            const auto u = quant(c.gu, normal(rng, (size_t) (H * FF), sd), H, FF);
            const auto d = quant(c.dn, normal(rng, (size_t) (H * FF), 1.0f / std::sqrt((float) FF)), FF, H);
            blobs.insert(blobs.end(), g.begin(), g.end());
            blobs.insert(blobs.end(), u.begin(), u.end());
            blobs.insert(blobs.end(), d.begin(), d.end());
        }
        if (blobs.size() != (size_t) NE * f.bytes) { std::printf("  %s: blob size  FAIL\n", c.what); ok = false; continue; }
        for (const int T : {1, 3, 8}) {
            Window w{T, {}};
            std::uniform_int_distribution<int> pick(0, NE - 1);
            for (int t = 0; t < T; ++t) {   // 1..NE distinct experts per token (top-k never repeats one), the
                                            // other ranks unrouted (-1: rows left alone by both sides)
                std::vector<int> perm = {0, 1, 2, 3, 4, 5};
                std::shuffle(perm.begin(), perm.end(), rng);
                const int m = 1 + pick(rng);
                for (int j = 0; j < K; ++j) w.route.push_back(j < m ? perm[(size_t) j] : -1);
            }
            const auto x = normal(rng, (size_t) T * H, 1.0f);
            const auto ref = run(f, blobs, x, w, 10.0f, nullptr), got = run(f, blobs, x, w, 10.0f, &pool);
            const auto ref0 = run(f, blobs, x, w, 0.0f, nullptr), got0 = run(f, blobs, x, w, 0.0f, &pool);
            const bool b1 = same(ref, got), b0 = same(ref0, got0), active = !same(ref, ref0);
            std::printf("  %-15s T=%d: pool vs one thread, clamp 10: %s; clamp 0 (Qwen's call): %s; clamp active: %s\n",
                        c.what, T, b1 ? "bitwise" : "DIFFERENT", b0 ? "bitwise" : "DIFFERENT", active ? "yes" : "NO");
            ok &= b1 && b0 && active;
        }
    }
    return ok;
}

bool combine_checks() {
    cudaStream_t st;
    GLM_CHECK(cudaStreamCreate(&st));
    std::mt19937 rng(77);
    const int E = 4096;
    bool ok = true;
    for (const int T : {1, 2, 5, 8}) {
        const auto parts = normal(rng, (size_t) T * K * E, 1.0f), shared = normal(rng, (size_t) T * E, 1.0f);
        const auto w = uniform(rng, (size_t) T * K, 0.0f, 2.5f);
        std::vector<int32_t> map((size_t) T * K, -1);
        std::vector<float> rows;
        int n = 0;
        for (int i = 0; i < T * K; ++i)
            if (rng() % 2) {   // a CPU row with the same floats, in a shuffled compact order
                map[(size_t) i] = n++;
                rows.insert(rows.end(), parts.begin() + (size_t) i * E, parts.begin() + (size_t) (i + 1) * E);
            }
        if (n == 0) { map[0] = 0; rows.assign(parts.begin(), parts.begin() + E); n = 1; }
        // the GPU's parts with the CPU rows' slots poisoned: the combine must not read them
        std::vector<float> poisoned = parts;
        for (int i = 0; i < T * K; ++i)
            if (map[(size_t) i] >= 0) std::fill(poisoned.begin() + (size_t) i * E, poisoned.begin() + (size_t) (i + 1) * E, 1e30f);
        Dev<float> dp(parts), dpp(poisoned), dw(w), ds(shared), dr(rows), o1((size_t) T * E), o2((size_t) T * E), o3((size_t) T * E);
        Dev<int32_t> dm(map);
        k::glm_moe_combine(dp.p, dw.p, ds.p, o1.p, E, K, T, st);
        k::glm_moe_combine(dpp.p, dw.p, ds.p, o2.p, E, K, T, st, dr.p, dm.p);
        auto rows2 = rows;
        for (auto& v : rows2) v *= 1.5f;
        dr.up(rows2);
        k::glm_moe_combine(dpp.p, dw.p, ds.p, o3.p, E, K, T, st, dr.p, dm.p);
        GLM_CHECK(cudaStreamSynchronize(st));
        const bool b = same(o1.down(), o2.down()), d = !same(o1.down(), o3.down());
        std::printf("  combine T=%d, %d of %d rows from the CPU: %s; other CPU rows change it: %s\n", T, n, T * K,
                    b ? "bitwise" : "DIFFERENT", d ? "yes" : "NO");
        ok &= b && d;
    }
    GLM_CHECK(cudaStreamDestroy(st));
    return ok;
}

}  // namespace

int main() {
    ggml_cpu_init();
    if (!kc::native_experts_available()) { std::printf("no native expert path: SKIP\n"); return 77; }
    bool ok = true;
    {
        kc::ExpertPool pool(4, /*pin=*/false, /*host_works=*/true);
        std::printf("[pool] run_split_multi_native, %d workers + the host\n", pool.workers());
        ok &= pool_checks(pool);
    }
    std::printf("[combine] glm_moe_combine with the CPU's rows\n");
    ok &= combine_checks();
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
