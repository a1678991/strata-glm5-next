// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// The widest expert of the architectures this engine reads: qwen4exp 2560 / 640, glm5-next 4096 / 2048.  The
/// pool's native buffers below are sized for these; a layer's own widths are its `NativeFmt::n_embd` / `n_ff`.
inline constexpr int64_t kNativeMaxEmbd = 4096;
inline constexpr int64_t kNativeMaxFF = 2048;
/// The largest activation size per 256 values among the vec_dot types of the expert formats with GPU kernels: Q8_K
/// (292 B; Q8_1 is 288, Q8_0 272).  F16/BF16/F32 experts have no GPU kernel, and native_fmt refuses them here.
inline constexpr size_t kNativeActPer256 = 292;
/// Bytes of the largest quantized activation any native layer uses: n_embd as Q8_K.  Qwen 2560: 2,920 (this was
/// 4096 when it was the only model); GLM 4096: 16 x 292 = 4,672 (a multiple of 64).
inline constexpr size_t kNativeActBytes = (size_t) kNativeMaxEmbd / 256 * kNativeActPer256;
/// Bytes of the largest quantized down activation: n_ff as Q8_K, rounded up to 64 B.  Qwen 640: Q8_0 680 (this was
/// 1024); GLM 2048: 8 x 292 = 2,336 -> 2,368.
inline constexpr size_t kNativeHBytes = ((size_t) kNativeMaxFF / 256 * kNativeActPer256 + 63) / 64 * 64;
static_assert(kNativeActBytes == 4672 && kNativeHBytes == 2368, "the native activation caps");
static_assert(kNativeActBytes >= 4096 && kNativeHBytes >= 1024, "never below the Qwen-only caps");

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace strata::kernels::cpu
