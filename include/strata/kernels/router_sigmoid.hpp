// include/strata/kernels/router_sigmoid.hpp - GLM-5.3-Flash's MoE router on the GPU: F32 logits, sigmoid gating with
// the exp_probs_b selection bias, top-k, normalized and scaled weights.  The semantics, line by line, are those of
// llama.cpp f1cee99 build_moe_ffn (src/llama-graph.cpp 2069-2173), transcribed in
// include/strata/kernels/cpu/router_sigmoid.hpp (the host version, also RouterLookahead's):
//
//   p = sigmoid(logits);  sel = p + bias;  ids = the k largest sel (descending; ties: lower index);
//   w = p[ids];  norm: w /= max(sum(w), 6.103515625e-5) (sum accumulated in double, in selection order);
//   w *= scale (when scale is neither 0 nor 1)
//
// All pointers are device memory; logits/ids/weights are [n_tok][...]; every call enqueues on `stream` without
// allocation or synchronization.
#pragma once

#include <cstdint>

namespace strata::kernels {

struct SigmoidRouterShape {
    int n_embd = 4096;
    int n_expert = 288;          ///< expert_count; at most 1024
    int k = 8;                   ///< expert_used_count; at most 32
    bool norm = true;            ///< expert_weights_norm
    float scale = 2.5f;          ///< expert_weights_scale
};

/// logits[t][e] = W[e] . x[t] for an F32 router (ffn_gate_inp, n_expert rows of n_embd; n_embd a multiple of 4).
/// One warp per expert row, the weights read once for every token of the call (any n_tok >= 1).
void router_f32_logits(const float* w, const float* x, float* logits, const SigmoidRouterShape& shape, int n_tok,
                       void* stream);
/// ids[t][0..k) and weights[t][0..k) from the logits; `bias` (n_expert, exp_probs_b) may be null.  One block per token.
void router_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* weights,
                         const SigmoidRouterShape& shape, int n_tok, void* stream);

}  // namespace strata::kernels
