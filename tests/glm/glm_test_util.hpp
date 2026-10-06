// tests/glm/glm_test_util.hpp - small helpers shared by the GLM-5.3-Flash kernel parity tests (tests/glm/*.cpp):
// device buffers, deterministic random data, the error statistics every test prints, and event timing.
//
// Error statistics, the same for every compared tensor:
//   max_abs  = max_i |gpu_i - ref_i|
//   rel      = max_abs / max_i |ref_i|           (the error relative to the tensor's scale; the PASS criterion)
//   rms_rel  = rms(gpu - ref) / rms(ref)
// A tolerance is always stated next to the number it bounds, with its reason in the test's header comment.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace glmtest {

#define GLM_CHECK(call)                                                                                              \
    do {                                                                                                             \
        const cudaError_t glm_e_ = (call);                                                                           \
        if (glm_e_ != cudaSuccess) {                                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(glm_e_));          \
            std::exit(2);                                                                                            \
        }                                                                                                            \
    } while (0)

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) { GLM_CHECK(cudaMalloc((void**) &p, std::max<size_t>(count, 1) * sizeof(T))); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& h) { GLM_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        GLM_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return h;
    }
    void zero() { GLM_CHECK(cudaMemset(p, 0, n * sizeof(T))); }
};

inline std::vector<float> normal(std::mt19937& rng, size_t n, float sd, float mean = 0.0f) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
inline std::vector<float> uniform(std::mt19937& rng, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

struct Err {
    double max_abs = 0, max_ref = 0, se = 0, sr = 0;
    size_t n = 0, nonfinite = 0;
    double rel() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
    double rms_rel() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
    void add(double a, double r) {
        if (!std::isfinite(a) || !std::isfinite(r)) { ++nonfinite; return; }
        const double d = std::fabs(a - r);
        max_abs = std::max(max_abs, d);
        max_ref = std::max(max_ref, std::fabs(r));
        se += d * d;
        sr += r * r;
        ++n;
    }
    void merge(const Err& o) {
        max_abs = std::max(max_abs, o.max_abs);
        max_ref = std::max(max_ref, o.max_ref);
        se += o.se;
        sr += o.sr;
        n += o.n;
        nonfinite += o.nonfinite;
    }
};

inline Err compare(const float* a, const float* r, size_t n) {
    Err e;
    for (size_t i = 0; i < n; ++i) e.add(a[i], r[i]);
    return e;
}

/// Prints one line and returns whether rel <= tol (and nothing was NaN/inf).
inline bool report(const char* name, const Err& e, double tol) {
    const bool ok = e.nonfinite == 0 && e.n > 0 && e.rel() <= tol;
    std::printf("  %-34s max_abs %.3e  max|ref| %.3e  rel %.3e  rms_rel %.3e  (tol %.1e)  %s\n", name, e.max_abs,
                e.max_ref, e.rel(), e.rms_rel(), tol, ok ? "PASS" : "FAIL");
    return ok;
}

/// Bitwise equality of two float arrays; prints the first difference.
inline bool bitwise(const char* name, const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t x, y;
        std::memcpy(&x, a + i, 4);
        std::memcpy(&y, b + i, 4);
        if (x != y) {
            std::printf("  %-34s bitwise: DIFFER at %zu (%.9g vs %.9g)  FAIL\n", name, i, a[i], b[i]);
            return false;
        }
    }
    std::printf("  %-34s bitwise identical (%zu values)  PASS\n", name, n);
    return true;
}

/// Median time in microseconds of `reps` calls of f (each bracketed by events on `stream`), after `warm` calls.
template <typename F>
double median_us(F&& f, cudaStream_t stream, int reps = 100, int warm = 10) {
    cudaEvent_t a, b;
    GLM_CHECK(cudaEventCreate(&a));
    GLM_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < warm; ++i) f();
    GLM_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> ms;
    for (int i = 0; i < reps; ++i) {
        GLM_CHECK(cudaEventRecord(a, stream));
        f();
        GLM_CHECK(cudaEventRecord(b, stream));
        GLM_CHECK(cudaEventSynchronize(b));
        float t = 0;
        GLM_CHECK(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(ms.begin(), ms.end());
    return 1000.0 * ms[ms.size() / 2];
}

inline bool has_flag(int argc, char** argv, const char* f) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], f) == 0) return true;
    return false;
}

}  // namespace glmtest
