// src/prefill/glm_gemm16.cu - the GLM prompt path's FP16 GEMM (glm::Gemm16, include/strata/prefill/glm_prefill_kernels.hpp):
// cublasGemmEx on CUDA, hipBLASLt's heuristic algorithm (else hipblasGemmEx) on HIP.  Host code only, in its own
// translation unit: hipBLASLt's headers declare the native HIP shuffles the CUDA-compatibility shim redefines (as in
// src/prefill/gemm.cu), which the kernels in glm_prefill_kernels.cu need.
#include "strata/prefill/glm_prefill_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#define STRATA_GLM_LT 1
#endif

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>

namespace strata::prefill::glm {

// ---- Gemm16 -----------------------------------------------------------------------------------------------------
#if defined(STRATA_GLM_LT)
namespace {
// One hipBLASLt plan per shape: the same column-major view as the cuBLAS call below (A = W^T, B = X, D = Y).
struct LtPlan {
    bool ok = false;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hipblasLtMatmulAlgo_t algo{};
};
struct LtState {
    hipblasLtHandle_t h = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 32u << 20;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>, LtPlan> plans;
    ~LtState() {
        for (auto& [k, p] : plans) {
            if (p.desc) hipblasLtMatmulDescDestroy(p.desc);
            if (p.la) hipblasLtMatrixLayoutDestroy(p.la);
            if (p.lb) hipblasLtMatrixLayoutDestroy(p.lb);
            if (p.lc) hipblasLtMatrixLayoutDestroy(p.lc);
        }
        if (ws) hipFree(ws);
        if (h) hipblasLtDestroy(h);
    }
    const LtPlan& plan(int64_t T, int64_t N, int64_t K, int64_t ldx, int64_t ldy) {
        auto [it, fresh] = plans.try_emplace({T, N, K, ldx, ldy});
        LtPlan& p = it->second;
        if (!fresh) return p;
        const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16F, (uint64_t) K, (uint64_t) N, K) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16F, (uint64_t) K, (uint64_t) T, ldx) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_32F, (uint64_t) N, (uint64_t) T, ldy) != HIPBLAS_STATUS_SUCCESS)
            return p;
        hipblasLtMatmulPreference_t pref = nullptr;
        if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) return p;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof ws_bytes);
        hipblasLtMatmulHeuristicResult_t res{};
        int n = 0;
        const hipblasStatus_t s = hipblasLtMatmulAlgoGetHeuristic(h, p.desc, p.la, p.lb, p.lc, p.lc, pref, 1, &res, &n);
        hipblasLtMatmulPreferenceDestroy(pref);
        if (s == HIPBLAS_STATUS_SUCCESS && n > 0) {
            p.algo = res.algo;
            p.ok = true;
        }
        return p;
    }
};
}  // namespace
#endif

Gemm16::~Gemm16() {
#if defined(STRATA_GLM_LT)
    delete static_cast<LtState*>(lt_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
}

bool Gemm16::init(void* stream, std::string& err, void* workspace, size_t ws_bytes) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "glm prefill gemm: cublasCreate: status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    if (cublasSetStream(h, (cudaStream_t) stream) != CUBLAS_STATUS_SUCCESS) {
        err = "glm prefill gemm: cublasSetStream failed";
        return false;
    }
    if (workspace != nullptr && ws_bytes > 0) cublasSetWorkspace(h, workspace, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
#if defined(STRATA_GLM_LT)
    const char* v = std::getenv("STRATA_GLM_GEMM_LT");
    if (v == nullptr || v[0] != '0') {
        auto* lt = new LtState;
        if (hipblasLtCreate(&lt->h) != HIPBLAS_STATUS_SUCCESS || hipMalloc(&lt->ws, lt->ws_bytes) != hipSuccess) {
            (void) hipGetLastError();
            delete lt;   // hipblasGemmEx only
        } else {
            lt_ = lt;
        }
    }
#endif
    return true;
}

void Gemm16::run(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t ldy, int64_t T, int64_t N,
                 int64_t K, float beta) {
    if (T <= 0 || N <= 0) return;
    const float alpha = 1.0f;
#if defined(STRATA_GLM_LT)
    if (lt_ != nullptr) {
        auto* lt = static_cast<LtState*>(lt_);
        const LtPlan& p = lt->plan(T, N, K, ldx, ldy);
        if (p.ok && hipblasLtMatmul(lt->h, p.desc, &alpha, W, p.la, X, p.lb, &beta, Y, p.lc, Y, p.lc, &p.algo, lt->ws,
                                    lt->ws_bytes, (hipStream_t) stream_) == HIPBLAS_STATUS_SUCCESS)
            return;
    }
#endif
    // column-major view: Y^T[N, T] = W[N, K] . X^T[K, T]; W is row-major [N][K] = column-major [K][N] transposed
    const cublasStatus_t s = cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,
                                          &alpha, W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) ldx, &beta, Y,
                                          CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "glm prefill gemm: GemmEx T=%lld N=%lld K=%lld: status %d\n", (long long) T, (long long) N,
                     (long long) K, (int) s);
        std::exit(1);
    }
}

}  // namespace strata::prefill::glm
