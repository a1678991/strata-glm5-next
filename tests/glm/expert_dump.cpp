// tests/glm/expert_dump.cpp - the Qwen bit-identity check of the GLM port's expert changes (the clamp epilogue and
// the Q6_K down rows in iq_kernels.cu, native_expert.cpp, iq_avx2/iq_avx512/kq_avx2.cpp).
//
// Writes, for fixed seeds, every output of the routed-expert paths a Qwen pack uses - the GPU's native_expert_grouped
// (default and STRATA_GROUPED_V1 kernels) and the CPU pool's native_quant_act / native_gu_rows / native_quant_h /
// native_down_rows (1..4 tokens per expert: ggml-cpu's dot at one token, the multi-token AVX-512/AVX-2/KQ kernels
// from two) - to one binary file, using only interfaces that exist unchanged in the base tree.  Built once against
// the base (the tree before the GLM changes) and once against this tree, the two files must be byte-identical:
//
//   expert_dump out.bin            (as is: AVX-512 where the CPU has it)
//   STRATA_NO_IQ512=1 expert_dump out_avx2.bin
//   STRATA_KQ256=1 expert_dump out_kq.bin
//
// Formats: the Qwen packs' (IQ3_S / IQ4_XS / IQ3_XXS / IQ2_S / Q4_K / Q5_K gate/up; IQ4_NL / Q8_0 / Q5_1 down) at
// 2560 x 640.  Random weights are quantized with ggml's quantizer (Strata's pinned ggml); every value is deterministic.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CK(call)                                                                                                     \
    do {                                                                                                             \
        const cudaError_t e_ = (call);                                                                               \
        if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); std::exit(2); }   \
    } while (0)

namespace k = strata::kernels;

namespace {

std::FILE* g_out = nullptr;
size_t g_bytes = 0;
void put(const void* p, size_t n) {
    std::fwrite(p, 1, n, g_out);
    g_bytes += n;
}

std::vector<uint8_t> quant(int type, std::mt19937& rng, int64_t n_in, int64_t rows, float sd) {
    std::normal_distribution<float> d(0.f, sd);
    std::vector<float> w((size_t) (n_in * rows));
    for (auto& v : w) v = d(rng);
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, n_in) * (size_t) rows);
    ggml_quantize_init((ggml_type) type);
    ggml_quantize_chunk((ggml_type) type, w.data(), q.data(), 0, rows, n_in, nullptr);
    return q;
}

void run(int gu, int dn, int64_t H, int64_t FF, unsigned seed) {
    std::mt19937 rng(seed);
    const k::NativeExpertLayout L = k::native_expert_layout(gu, dn, H, FF);
    constexpr int E = 6, T = 4;
    // the routing: expert e's tokens
    const std::vector<std::vector<int>> toks = {{0, 1, 2, 3}, {1}, {0, 2}, {3}, {0, 1, 2}, {2}};
    std::vector<uint8_t> blobs;
    for (int e = 0; e < E; ++e) {
        const auto g = quant(gu, rng, H, FF, 0.05f), u = quant(gu, rng, H, FF, 0.05f), dw = quant(dn, rng, FF, H, 0.05f);
        blobs.insert(blobs.end(), g.begin(), g.end());
        blobs.insert(blobs.end(), u.begin(), u.end());
        blobs.insert(blobs.end(), dw.begin(), dw.end());
    }
    if (blobs.size() != (size_t) E * L.bytes) { std::fprintf(stderr, "blob size\n"); std::exit(2); }
    std::normal_distribution<float> xd(0.f, 1.f);
    std::vector<float> x((size_t) (T * H));
    for (auto& v : x) v = xd(rng);

    // ---- GPU
    std::vector<int32_t> start = {0}, ent_tok, ent_dst;
    for (int e = 0; e < E; ++e) {
        for (int t : toks[(size_t) e]) { ent_tok.push_back(t); ent_dst.push_back((int32_t) ent_dst.size()); }
        start.push_back((int32_t) ent_tok.size());
    }
    const int n_ent = (int) ent_tok.size();
    uint8_t* d_blobs;
    float *d_x, *d_out;
    void *d_xq, *d_scr;
    int32_t *d_start, *d_ng, *d_tok, *d_dst;
    unsigned long long* d_ptr;
    CK(cudaMalloc((void**) &d_blobs, blobs.size()));
    CK(cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &d_x, x.size() * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xq, (size_t) T * (H / 32) * 36));
    CK(cudaMalloc(&d_scr, k::native_expert_scratch_bytes(n_ent, FF)));
    CK(cudaMalloc((void**) &d_out, (size_t) n_ent * H * 4));
    std::vector<unsigned long long> ptr;
    for (int e = 0; e < E; ++e) ptr.push_back((unsigned long long) (d_blobs + (size_t) e * L.bytes));
    CK(cudaMalloc((void**) &d_ptr, ptr.size() * 8));
    CK(cudaMemcpy(d_ptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &d_start, start.size() * 4));
    CK(cudaMemcpy(d_start, start.data(), start.size() * 4, cudaMemcpyHostToDevice));
    const int32_t ng = E;
    CK(cudaMalloc((void**) &d_ng, 4));
    CK(cudaMemcpy(d_ng, &ng, 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &d_tok, ent_tok.size() * 4));
    CK(cudaMemcpy(d_tok, ent_tok.data(), ent_tok.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &d_dst, ent_dst.size() * 4));
    CK(cudaMemcpy(d_dst, ent_dst.data(), ent_dst.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    k::quantize_q8_1_rows(d_x, T, H, d_xq, st);
    std::vector<float> out((size_t) n_ent * H);
    for (int v1 = 0; v1 < 2; ++v1) {
        k::native_grouped_set_v1(v1 != 0);
        CK(cudaMemset(d_out, 0, out.size() * 4));
        k::native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, E, n_ent, d_xq, d_scr, d_out, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
        put(out.data(), out.size() * 4);
    }
    k::native_grouped_set_v1(false);
    cudaFree(d_blobs); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_scr); cudaFree(d_out); cudaFree(d_ptr);
    cudaFree(d_start); cudaFree(d_ng); cudaFree(d_tok); cudaFree(d_dst);
    cudaStreamDestroy(st);

    // ---- CPU pool kernels
    k::cpu::NativeFmt f;
    std::string err;
    if (!k::cpu::native_fmt(gu, dn, H, FF, f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); std::exit(2); }
    std::vector<std::vector<uint8_t>> act(T, std::vector<uint8_t>(f.act_bytes));
    for (int t = 0; t < T; ++t) {
        k::cpu::native_quant_act(f, x.data() + (size_t) t * H, act[(size_t) t].data());
        put(act[(size_t) t].data(), act[(size_t) t].size());
    }
    for (int e = 0; e < E; ++e) {
        const uint8_t* blob = blobs.data() + (size_t) e * L.bytes;
        const int nt = (int) toks[(size_t) e].size();
        std::vector<const void*> a;
        std::vector<std::vector<float>> ffv((size_t) nt, std::vector<float>((size_t) FF)), ov((size_t) nt,
                                                                                             std::vector<float>((size_t) H));
        std::vector<float*> ffp, op;
        std::vector<std::vector<uint8_t>> hq((size_t) nt, std::vector<uint8_t>(f.h_bytes));
        std::vector<const void*> hp;
        for (int j = 0; j < nt; ++j) {
            a.push_back(act[(size_t) toks[(size_t) e][(size_t) j]].data());
            ffp.push_back(ffv[(size_t) j].data());
            op.push_back(ov[(size_t) j].data());
        }
        k::cpu::native_gu_rows(f, blob, a.data(), nt, ffp.data(), 0, (int) FF);
        for (int j = 0; j < nt; ++j) {
            put(ffv[(size_t) j].data(), ffv[(size_t) j].size() * 4);
            k::cpu::native_quant_h(f, ffv[(size_t) j].data(), hq[(size_t) j].data());
            hp.push_back(hq[(size_t) j].data());
        }
        k::cpu::native_down_rows(f, blob, hp.data(), nt, op.data(), 0, (int) H);
        for (int j = 0; j < nt; ++j) put(ov[(size_t) j].data(), ov[(size_t) j].size() * 4);
    }
    std::printf("  gate/up %2d down %2d  %lld x %lld: %d GPU entries, CPU 1..4 tokens per expert\n", gu, dn,
                (long long) H, (long long) FF, n_ent);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: expert_dump out.bin\n"); return 2; }
    g_out = std::fopen(argv[1], "wb");
    if (!g_out) { std::perror(argv[1]); return 2; }
    // the Qwen packs' formats (Strata's native_experts.txt): IQ3_S/IQ4_NL (43 layers of the UD-IQ4_XS pack),
    // IQ3_S/Q8_0, IQ4_XS/Q8_0, the Coder / IQ2 packs' IQ3_XXS and IQ2_S, Unsloth's Q4_K/Q5_1 and Q5_K/Q8_0
    const int cfg[][2] = {{21, 20}, {21, 8}, {23, 8}, {18, 20}, {22, 20}, {12, 7}, {13, 8}};
    unsigned seed = 11;
    for (const auto& c : cfg) run(c[0], c[1], 2560, 640, seed++);
    std::fclose(g_out);
    std::printf("expert_dump: %zu bytes to %s\n", g_bytes, argv[1]);
    return 0;
}
