// include/strata/kernels/cpu/router_sigmoid.hpp - the sigmoid MoE router with a selection bias (DeepSeek-V3 style,
// GLM-5.3-Flash: expert_gating_func 2, exp_probs_b, expert_weights_norm, expert_weights_scale 2.5) for one token on
// the host.  Header-only, no ggml: RouterLookahead (src/core/expert_source.cpp) can call it to predict the next
// layer's experts - for GLM it must add exp_probs_b, or it warms the wrong experts - and the GPU router's parity test
// uses it as its scalar reference.
//
// Transcribed from llama.cpp f1cee99 src/llama-graph.cpp, llm_graph_context::build_moe_ffn:
//   2069       probs = ggml_sigmoid(logits)                              1 / (1 + expf(-x)), ggml_vec_sigmoid_f32
//   2086-2089  selection_probs = probs + exp_probs_b   (probs stay unbiased: they give the weights)
//   2131       selected = ggml_argsort_top_k(selection_probs, k)        a descending argsort, first k (ggml.c 5450)
//   2145       weights = ggml_get_rows(probs, selected)                 in selection order
//   2156-2168  norm_w: sum = ggml_sum_rows(weights) (ggml_vec_sum_f32: a double accumulation, rounded to float);
//              sum = clamp(sum, 6.103515625e-5, INF); weights = weights / sum
//   2170-2173  w_scale != 0 && != 1: weights = weights * w_scale
// n_expert_groups is 1 for GLM-5.3-Flash (expert_group_count), so the group masking of 2105-2127 is not taken.
// Ties in selection_probs choose the lower expert index here (ggml's std::sort leaves their order unspecified).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace strata::kernels::cpu {

/// One token: logits (n_expert) -> ids (k, descending selection order) and, if `weights` is non-null, their weights.
/// `bias` may be null (no selection bias).  `scratch` is reused between calls (n_expert entries).
inline void sigmoid_route(const float* logits, const float* bias, int n_expert, int k, bool norm, float scale,
                          int32_t* ids, float* weights, std::vector<float>& probs, std::vector<int32_t>& order) {
    probs.resize((size_t) n_expert);
    order.resize((size_t) n_expert);
    std::vector<float> sel((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        probs[(size_t) e] = 1.f / (1.f + std::exp(-logits[e]));
        sel[(size_t) e] = bias ? probs[(size_t) e] + bias[e] : probs[(size_t) e];
        order[(size_t) e] = e;
    }
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
        const float va = sel[(size_t) a], vb = sel[(size_t) b];
        return va > vb || (va == vb && a < b);
    });
    for (int j = 0; j < k; ++j) ids[j] = order[(size_t) j];
    if (!weights) return;
    for (int j = 0; j < k; ++j) weights[j] = probs[(size_t) ids[j]];
    if (norm) {
        double acc = 0.0;
        for (int j = 0; j < k; ++j) acc += (double) weights[j];
        float sum = (float) acc;
        sum = std::max(sum, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) weights[j] = weights[j] / sum;
    }
    if (scale != 0.0f && scale != 1.0f)
        for (int j = 0; j < k; ++j) weights[j] = weights[j] * scale;
}

/// Selection only (the lookahead's question: which experts), the same ids as sigmoid_route.
inline void sigmoid_select(const float* logits, const float* bias, int n_expert, int k, int32_t* ids,
                           std::vector<float>& probs, std::vector<int32_t>& order) {
    sigmoid_route(logits, bias, n_expert, k, false, 1.0f, ids, nullptr, probs, order);
}

}  // namespace strata::kernels::cpu
