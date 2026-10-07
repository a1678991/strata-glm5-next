// include/strata/kernels/mhc.hpp - GLM-5.3-Flash's manifold-constrained hyper-connections (mHC, DeepSeek V4's
// "hc"): the residual stream is hc = 4 streams of n_embd, mixed before and after every attention/FFN block.
//
// Reference: llama.cpp f1cee99 src/models/glm5-next.cpp build_hc_pre (G:454-510), build_hc_sinkhorn (G:417-452),
// build_hc_post (G:512-547), build_hc_mean (G:370-378); with the fused ops llama.cpp enables by default
// (llama-context.cpp:240-242): ggml_dsv4_hc_comb / _pre / _post, ggml-cpu ops.cpp 11337-11645.  Per token t:
//
//   flat    = rms_norm(x[t] as one hc * n_embd vector, rms_eps)                         mhc_flat_norm
//   mixes   = hc_fn . flat                     ((2 + hc) * hc = 24 values: a native Q8_0 GEMV, the caller's)
//   pre[h]  = sigmoid(mixes[h] * scale[0] + base[h]) + hc_eps                          mhc_pre
//   post[h] = 2 sigmoid(mixes[hc + h] * scale[1] + base[hc + h])                       mhc_pre
//   comb    = Sinkhorn(mixes[2hc + .] * scale[2] + base[2hc + .]), hc x hc, n_iter     mhc_pre
//   cur[i]  = sum_h x[t][h][i] * pre[h]                       (the block's input)      mhc_pre
//   out[d][i] = f[i] * post[d] + sum_s residual[s][i] * comb[d + hc s]                 mhc_post (after the block)
//   head[i] = (((x[0][i] + x[1][i]) + x[2][i]) + x[3][i]) * (1 / hc)                   mhc_mean (before output_norm)
//
// Sinkhorn exactly as ggml_compute_forward_dsv4_hc_comb_f32: for each src a softmax over dst, then * (1/sum) + eps;
// one normalization over src ("cols"); then n_iter - 1 times rows (over dst) and cols, every sum starting at eps and
// applied as a multiplication by its reciprocal.  comb is stored [t][dst + hc * src], ggml's [dst, src, t].
//
// Layouts: x / residual / out [T][hc][n_embd]; cur / f / head [T][n_embd]; mixes [T][24]; post [T][hc];
// comb [T][hc * hc].  All F32 device memory, every call enqueues on `stream`, no allocation, no synchronization.
// hc must be 4 (G asserts it; the ggml op hard-codes it); the other dimensions are runtime arguments.
#pragma once

namespace strata::kernels {

struct MhcShape {
    int n_embd = 4096;
    int hc = 4;                 ///< hyper_connection.count; must be 4
    float rms_eps = 1e-5f;      ///< attention.layer_norm_rms_epsilon (the flat RMS norm, G:469)
    float hc_eps = 1e-6f;       ///< hyper_connection.epsilon
    int sinkhorn_iters = 20;    ///< hyper_connection.sinkhorn_iterations
    int mix_dim() const { return (2 + hc) * hc; }
};

/// flat[t] = x[t] / sqrt(mean(x[t]^2) + rms_eps) over all hc * n_embd values (no weight).  One block per token.
void mhc_flat_norm(const float* x, float* flat, const MhcShape& shape, int n_tok, void* stream);
/// pre/post/comb from the mixes and cur = sum_h x[h] * pre[h].  post and comb are the inputs of mhc_post.
void mhc_pre(const float* x, const float* mixes, const float* hc_scale, const float* hc_base, float* cur, float* post,
             float* comb, const MhcShape& shape, int n_tok, void* stream);
/// out = f * post + comb-mixed residual.  out may alias residual (each thread reads its hc residual values first).
void mhc_post(const float* f, const float* residual, const float* post, const float* comb, float* out,
              const MhcShape& shape, int n_tok, void* stream);
/// The final collapse: the plain mean of the hc streams, summed in stream order as ggml_add's chain.
void mhc_mean(const float* x, float* out, const MhcShape& shape, int n_tok, void* stream);

}  // namespace strata::kernels
