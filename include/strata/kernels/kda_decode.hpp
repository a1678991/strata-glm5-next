// include/strata/kernels/kda_decode.hpp - GLM-5.3-Flash (llama.cpp arch glm5-next): the decode-step kernels of a KDA
// (Kimi delta attention) layer, for one token or a verify window of up to kVerifyMaxT tokens.
//
// Reference: llama.cpp f1cee99, src/models/glm5-next.cpp build_kda_layer (G:686-762) and glm5_conv1d (G:196-235);
// ggml-cpu ops.cpp ggml_compute_forward_gated_delta_net_one_chunk, KDA mode (g has S_v values per head, ops.cpp
// 11124-11275).  The layer, with every projection a plain GEMV (native_mmvq, Q8_0 in the GGUF) done by the caller:
//
//   q, k, v = conv4(W_q x), conv4(W_k x), conv4(W_v x), each followed by SiLU     (three depthwise causal convs)
//   q, k    = q / sqrt(|q|^2 + 1e-6), the same for k, per 128-wide head          (ggml: rms_norm(eps/n) / sqrt(n))
//   g       = lb * sigmoid(-(A[h] * (f_b(f_a x) + dt_b)))       per CHANNEL (lb = gate_lower_bound = -5;
//                                                              ssm_a is already -exp(A_log): not exponentiated again)
//   beta    = sigmoid(W_beta x)                                 per head
//   S       = diag(exp(g)) S;  S += k (beta (v - S^T k))^T;  o = S^T q / sqrt(128)     (decay along the KEY index)
//   y       = RMSNorm(o) * ssm_norm * sigmoid(g_b(g_a x))       per head (eps = attention.layer_norm_rms_epsilon)
//   out     = W_o y
//
// Layouts (all F32, all device memory):
//   qkv      [T][3 * d_inner]   the raw q | k | v projections (d_inner = n_head * head_dim = 8192)
//   history  [3 * d_inner][3]   the conv state, q channels then k then v, the 3 previous inputs per channel, oldest
//                               first: llama.cpp's conv-state layout (glm5_conv1d's three [d_conv-1, d_inner] views)
//   conv_w   [d_inner][4]       per part (q, k, v): the GGUF's ssm_conv1d_{q,k,v} [4, 1, d_inner] as stored
//   h        [T][3 * d_inner]   conv + SiLU output, q and k heads L2-normalized, v as is
//   state    [n_head][128][128] state[(h * 128 + i) * 128 + j] = S[i][j], i the key index, j the value index
//                               (llama.cpp keeps the transpose per head: s[(h * 128 + j) * 128 + i])
//   fb       [T][d_inner]       f_b(f_a x), WITHOUT dt_b (the kernel adds ssm_dt.bias)
//   b_raw    [T][n_head]        W_beta x, before the sigmoid
//   gb       [T][d_inner]       g_b(g_a x), before the sigmoid
//   y        [T][d_inner]       the gated, normalized output (the W_o GEMV's input)
//
// Verify windows (Verifier, kVerifyMaxT = 8), the same split as the GDN kernels in verify_kernels.hpp: the verify
// half reads the conv history and the state and writes only the T outputs; the commit half, launched once the
// accepted count is known, replays the first *n_keep tokens from the h/fb/b_raw buffers the verify half used and
// writes history and state.  A rejected draft therefore never touches the state, and the state after any prefix of
// the window is recoverable without a snapshot of the 4 MiB-per-layer state.  The replay runs the per-token
// arithmetic of the single-token kernel, so the committed state is bitwise the state of n_keep single steps.
//
// Every dimension is a runtime argument; the kernels require head_dim == 128 (the S=128 tiling of fused_gdn) and
// throw std::invalid_argument otherwise.  All calls enqueue on `stream` and neither allocate nor synchronize, so
// they can be captured in a graph; n_keep is read from device memory for that reason.
#pragma once

#include <cstdint>

namespace strata::kernels {

struct KdaShape {
    int n_head = 64;                  ///< KDA heads (attention.head_count)
    int head_dim = 128;               ///< kda.head_dim; must be 128
    float gate_lower_bound = -5.0f;   ///< kda.gate_lower_bound (finite: the sigmoid form of G:714-718)
    float rms_eps = 1e-5f;            ///< attention.layer_norm_rms_epsilon, for the per-head output RMSNorm
    float l2_eps = 1e-6f;             ///< the L2 norm's eps (G:738, "Match FLA l2 norm")
    int d_inner() const { return n_head * head_dim; }
};

/// The three causal convs + SiLU + the q/k L2 norm for `n_tok` tokens (1..kVerifyMaxT).
/// update_history = true: the decode step - history afterwards holds the last 3 inputs of [history | qkv_0..].
/// update_history = false: the verify half - history is only read.
void kda_conv_l2(float* history, const float* qkv, const float* conv_wq, const float* conv_wk, const float* conv_wv,
                 float* h, const KdaShape& shape, int n_tok, bool update_history, void* stream);
/// The commit half of the conv: history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (device
/// memory; 0 leaves it unchanged).
void kda_conv_commit(float* history, const float* qkv, const KdaShape& shape, const int32_t* n_keep, void* stream);

/// The recurrence with its gates and the output norm for `n_tok` tokens (1..kVerifyMaxT), one block per head.
/// write_state = true: the decode step - y for every token, state written after the last.
/// write_state = false: the verify half - y for every token, state only read.
void kda_step_norm(float* state, const float* h, const float* fb, const float* dt_bias, const float* ssm_a,
                   const float* b_raw, const float* gb, const float* o_norm_w, float* y, const KdaShape& shape,
                   int n_tok, bool write_state, void* stream);
/// The commit half of the recurrence: replays the first *n_keep (device memory) tokens and writes the state; no y.
void kda_step_commit(float* state, const float* h, const float* fb, const float* dt_bias, const float* ssm_a,
                     const float* b_raw, const KdaShape& shape, const int32_t* n_keep, void* stream);

}  // namespace strata::kernels
