// include/strata/core/glm_pool.hpp - GLM-5.3-Flash's RAM-tier experts on the CPU expert pool: an expert that lives in the page-locked RAM tier (FileExpertSource::pin_cache_complement) is computed by
// the CPU pool (kernels::cpu::ExpertPool::run_split_multi_native with the clamped SwiGLU) instead of being copied over
// PCIe into VRAM staging and computed on the GPU.  The RAM bytes are read once, by the CPU, instead of RAM read + PCIe
// write + VRAM read; the GPU computes the VRAM-cached experts and (for now) the file-tier ones meanwhile.
//
// One layer call, driven by GlmExperts::moe:
//   begin(layer, T)            the window's tokens; every entry's map slot -1 (the GPU's row)
//   add(blob, entries, n)      one RAM-tier expert and its routed entries (row t * k + j); one pool job, nt tokens
//   run(x)                     ggml-cpu's activation of every token with a CPU entry (the layer's vec_dot_type: Q8_K for
//                              IQ3_S / IQ4_XS gate/up), then the pool's three phases (gate/up rows with the clamp, the
//                              intermediate's quantization, down rows) into compact rows
//   upload_src / upload_bytes  [map: cap int32, padded to 256 B][rows: n x n_embd f32], one host-to-device copy; the
//                              combine (glm_moe_combine's cpu_rows / cpu_map) reads entry t * k + j from rows[map[...]]
//
// Arithmetic: ggml-cpu's (Strata's pinned ggml, whose IQ / K-quant dot products are llama.cpp f1cee99's CPU ones): the
// activation is ggml's Q8_K (blocks of 256, an f32 scale) where the GPU kernels quantize to Q8_1 (blocks of 32, f16
// scales), so a CPU expert's row is not the GPU's row bit for bit; it is what llama.cpp's CPU backend computes for the
// same expert.  From two tokens per expert the AVX-512 / AVX-2 multi-token gate/up kernels round on their own (#152;
// STRATA_IQ_MT_MIN=1 makes a single token take them too, so the output no longer depends on the window's grouping).
#pragma once

#include "strata/core/glm.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class GlmPool {
public:
    struct Stats {
        int64_t calls = 0;        ///< layer calls with CPU work
        int64_t experts = 0;      ///< distinct experts the CPU computed (per layer call)
        int64_t entries = 0;      ///< routed (token, expert) entries the CPU served
        uint64_t bytes = 0;       ///< expert bytes the CPU read
        double ms_quant = 0;      ///< activation quantization
        double ms_run = 0;        ///< the pool (gate/up, intermediate, down)
    };

    GlmPool() = default;
    ~GlmPool();
    GlmPool(const GlmPool&) = delete;
    GlmPool& operator=(const GlmPool&) = delete;

    /// `pool` runs the work (not owned).  Every MoE layer's native format (kernels::cpu::expert_layout()) must have
    /// ggml-cpu dot products and fit the pool's buffers.  `max_t` tokens, `k` experts per token.
    bool init(const ModelGeometry& g, strata::kernels::cpu::ExpertPool* pool, int max_t, int k, std::string& err);
    bool ready() const { return pool_ != nullptr; }

    /// Pinned host rows for the window's activations, [max_t][n_embd]: the caller copies x here before run().
    float* host_x() const { return h_x_; }

    void begin(int64_t layer, int T);
    /// One expert of this layer for the CPU: its blob (stable until run() returns) and its entries (t * k + j).
    void add(const uint8_t* blob, const int32_t* entries, int n);
    int jobs() const { return njobs_; }
    int rows() const { return nrows_; }
    /// The pool's work on host_x(); blocks until the rows are written.
    void run();

    /// What the device needs: the map, then the rows (upload_bytes() of them).
    const void* upload_src() const { return h_buf_; }
    size_t upload_bytes() const { return map_bytes_ + (size_t) nrows_ * (size_t) E_ * sizeof(float); }
    size_t map_bytes() const { return map_bytes_; }
    /// The device block's size for max_t tokens of k experts: map and max_t * k rows.
    static size_t device_bytes(int max_t, int k, int64_t n_embd);

    const Stats& stats() const { return st_; }
    void reset_stats() { st_ = Stats{}; }   ///< after a calibration (GlmExperts::calibrate_pool)
    int workers() const;

private:
    strata::kernels::cpu::ExpertPool* pool_ = nullptr;
    std::vector<strata::kernels::cpu::NativeFmt> fmt_;   ///< per layer
    float limit_ = 0.0f;                                 ///< swiglu_clamp_exp
    int max_t_ = 0, k_ = 0;
    int64_t E_ = 0;
    int64_t layer_ = -1;
    int T_ = 0;
    size_t map_bytes_ = 0;
    void* h_buf_ = nullptr;      ///< pinned: map [cap] int32 (padded) | rows [cap][n_embd] f32
    float* h_x_ = nullptr;       ///< pinned: [max_t][n_embd]
    int32_t* map_ = nullptr;
    float* rows_ = nullptr;
    std::vector<uint8_t> nact_;  ///< [max_t][kNativeActBytes]: ggml's activation per token
    std::vector<uint8_t> tok_used_;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_;
    int njobs_ = 0, nrows_ = 0;
    Stats st_;
};

}  // namespace strata::core
