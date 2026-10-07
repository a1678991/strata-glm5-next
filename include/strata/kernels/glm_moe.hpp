// include/strata/kernels/glm_moe.hpp - GLM-5.3-Flash's MoE output: the routed experts' rows weighted and summed, plus
// the shared expert, in llama.cpp f1cee99's order (build_moe_ffn, src/llama-graph.cpp 2342-2370, then G:657):
//
//   moe[t]  = ((parts[t][0] * w[t][0] + parts[t][1] * w[t][1]) + parts[t][2] * w[t][2]) + ...    (ggml_mul, ggml_add)
//   out[t]  = moe[t] + shared[t]                                                                 (ffn_moe_out + ffn_shexp)
//
// Every product and sum is rounded to f32 on its own (no fused multiply-add), as the separate ggml ops do.
// parts [T][k][n_embd], w [T][k], shared / out [T][n_embd]: device F32, the caller's stream, no allocation.
#pragma once

namespace strata::kernels {

void glm_moe_combine(const float* parts, const float* w, const float* shared, float* out, int n_embd, int k, int n_tok,
                     void* stream);

}  // namespace strata::kernels
