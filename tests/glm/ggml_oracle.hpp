// tests/glm/ggml_oracle.hpp - running ggml-cpu at llama.cpp f1cee99 (the GLM reference) from a parity test.
//
// The tests link the CPU-only shared ggml of that checkout (STRATA_GLM_ORACLE_BUILD_DIR, built by
// tools/glm/build_oracle.sh), not the static ggml 3cf0325 Strata's CPU experts use: the oracle is the commit whose
// src/models/glm5-next.cpp defines the model.  Nothing else in the test executables links a ggml.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace glmtest {

struct Oracle {
    ggml_context* ctx = nullptr;
    int threads = 16;
    explicit Oracle(size_t mem_bytes, int n_threads = 16) : threads(n_threads) {
        ggml_init_params p{};
        p.mem_size = mem_bytes;
        p.mem_buffer = nullptr;
        p.no_alloc = false;
        ctx = ggml_init(p);
        if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); std::exit(2); }
    }
    ~Oracle() { ggml_free(ctx); }
    Oracle(const Oracle&) = delete;
    Oracle& operator=(const Oracle&) = delete;

    ggml_tensor* f32(const std::vector<float>& v, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        if ((size_t) ggml_nelements(t) != v.size()) { std::fprintf(stderr, "oracle: size mismatch\n"); std::exit(2); }
        std::memcpy(t->data, v.data(), v.size() * sizeof(float));
        return t;
    }
    /// A weight of `rows` rows of `n_in` values in `type`, from quantized bytes (ggml_quantize_chunk's output).
    ggml_tensor* weight(ggml_type type, const std::vector<uint8_t>& bytes, int64_t n_in, int64_t rows) {
        ggml_tensor* t = ggml_new_tensor_2d(ctx, type, n_in, rows);
        if (ggml_nbytes(t) != bytes.size()) { std::fprintf(stderr, "oracle: weight size mismatch\n"); std::exit(2); }
        std::memcpy(t->data, bytes.data(), bytes.size());
        return t;
    }
    void compute(ggml_tensor* out) { compute(std::vector<ggml_tensor*>{out}); }
    void compute(const std::vector<ggml_tensor*>& outs) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
        for (ggml_tensor* o : outs) ggml_build_forward_expand(gf, o);
        if (ggml_graph_compute_with_ctx(ctx, gf, threads) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "oracle: graph compute failed\n");
            std::exit(2);
        }
    }
    static std::vector<float> read(const ggml_tensor* t) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { std::fprintf(stderr, "oracle: read needs contiguous F32\n"); std::exit(2); }
        std::vector<float> v((size_t) ggml_nelements(t));
        std::memcpy(v.data(), t->data, v.size() * sizeof(float));
        return v;
    }
};

/// Quantizes `rows` rows of `n_in` floats with ggml's own quantizer (the bytes a GGUF of that type holds).
inline std::vector<uint8_t> quantize(ggml_type type, const std::vector<float>& w, int64_t n_in, int64_t rows) {
    std::vector<uint8_t> q(ggml_row_size(type, n_in) * (size_t) rows);
    ggml_quantize_init(type);
    const size_t got = ggml_quantize_chunk(type, w.data(), q.data(), 0, rows, n_in, nullptr);
    if (got != q.size()) { std::fprintf(stderr, "quantize: %zu of %zu bytes\n", got, q.size()); std::exit(2); }
    return q;
}

}  // namespace glmtest
