// tests/dsa/dsa_oracle.cpp - see dsa_oracle.hpp.
#include "dsa_oracle.hpp"

#include "strata/kernels/f16_bits.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dsa_test {

namespace k = strata::kernels;

namespace {

struct Ctx {
    ggml_context* c = nullptr;
    Ctx(size_t mem, bool no_alloc) {
        ggml_init_params p{mem, nullptr, no_alloc};
        c = ggml_init(p);
        if (!c) { std::fprintf(stderr, "oracle: ggml_init(%zu) failed\n", mem); std::exit(2); }
    }
    ~Ctx() { ggml_free(c); }
    operator ggml_context*() const { return c; }
};

// A tensor over caller-owned host memory (the context holds only metadata).
ggml_tensor* wrap(ggml_context* ctx, ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2, const void* data) {
    ggml_tensor* x = ggml_new_tensor_3d(ctx, t, ne0, ne1, ne2);
    x->data = const_cast<void*>(data);
    return x;
}
ggml_tensor* input_f32(ggml_context* ctx, const float* data, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1,
                       int64_t ne3 = 1) {
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    std::memcpy(x->data, data, ggml_nbytes(x));
    return x;
}
void compute(ggml_context* ctx, std::initializer_list<ggml_tensor*> outs, int n_threads) {
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);
    for (ggml_tensor* t : outs) ggml_build_forward_expand(gf, t);
    if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "oracle: graph compute failed\n");
        std::exit(2);
    }
}
void read(std::vector<float>& dst, const ggml_tensor* t) {
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { std::fprintf(stderr, "oracle: read of a non-f32/cont\n"); std::exit(2); }
    dst.resize((size_t) ggml_nelements(t));
    std::memcpy(dst.data(), t->data, ggml_nbytes(t));
}

std::vector<uint8_t> q8_0(const std::vector<float>& w, int64_t rows, int64_t n_per_row) {
    std::vector<uint8_t> out(ggml_row_size(GGML_TYPE_Q8_0, n_per_row) * rows);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), out.data(), 0, rows, n_per_row, nullptr);
    return out;
}
std::vector<float> gauss(std::mt19937_64& rng, size_t n, float sd, float mean = 0.0f) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

uint32_t kpool_pad(uint32_t n_pool) {   // llama-memory-hybrid-idx.cpp:386
    const uint32_t p = ((n_pool + 1 + 63) / 64) * 64;
    return std::max<uint32_t>(64u, p);
}

}  // namespace

float f16_round(float x) { return k::f32_from_f16(k::f16_from_f32(x)); }

void quantize_latent_row(const float* x, int n, int8_t* codes, uint16_t* scales) {
    for (int g = 0; g < n / k::DSA_LAT_GROUP; ++g) {
        float amax = 0.0f;
        for (int j = 0; j < k::DSA_LAT_GROUP; ++j) amax = std::fmax(amax, std::fabs(x[g * k::DSA_LAT_GROUP + j]));
        const uint16_t sbits = k::f16_from_f32(amax / 127.0f);
        const float sf = k::f32_from_f16(sbits);
        scales[g] = sbits;
        for (int j = 0; j < k::DSA_LAT_GROUP; ++j) {
            int q = 0;
            if (sf > 0.0f) {
                q = (int) std::nearbyint(x[g * k::DSA_LAT_GROUP + j] / sf);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
            }
            codes[g * k::DSA_LAT_GROUP + j] = (int8_t) q;
        }
    }
}

HostWeights make_weights(const DsaShapes& s, uint64_t seed) {
    std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + 11);
    HostWeights w;
    auto mat = [&](int64_t n_in, int64_t n_out) {
        return q8_0(gauss(rng, (size_t) (n_in * n_out), 1.0f / std::sqrt((float) n_in)), n_out, n_in);
    };
    w.wq_a = mat(s.n_embd, s.q_lora);
    w.wq_b = mat(s.q_lora, s.n_head * s.head_k);
    w.wkv_a = mat(s.n_embd, s.kv_lora);
    w.wk_b = mat(s.head_k, s.n_head * s.kv_lora);   // [n_head][kv_lora rows][head_k]
    w.wv_b = mat(s.kv_lora, s.n_head * s.head_v);   // [n_head][head_v rows][kv_lora]
    w.wo = mat(s.n_head * s.head_v, s.n_embd);
    w.idx_q_b = mat(s.q_lora, s.idx_n_head * s.idx_dim);
    w.idx_k = mat(s.n_embd, s.idx_dim);
    w.idx_gate = mat(s.n_embd, s.idx_dim);
    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    auto pos = [&](size_t n) { std::vector<float> v(n); for (float& x : v) x = u(rng); return v; };
    w.q_a_norm = pos((size_t) s.q_lora);
    w.kv_a_norm = pos((size_t) s.kv_lora);
    w.idx_k_norm = pos((size_t) s.idx_dim);
    w.idx_k_norm_b = gauss(rng, (size_t) s.idx_dim, 0.1f);
    w.idx_ape = gauss(rng, (size_t) (s.kpool * s.idx_dim), 0.5f);
    w.idx_proj = gauss(rng, (size_t) (s.n_embd * s.idx_n_head), 1.0f / std::sqrt((float) s.n_embd));
    return w;
}

void cache_init(HostCache& c, const DsaShapes& s, int fmt, int64_t max_cells) {
    c.fmt = fmt;
    c.n_cells = 0;
    c.latent.assign((size_t) (max_cells * s.kv_lora), 0.0f);
    c.codes.assign(fmt == k::kDsaLatInt8 ? (size_t) (max_cells * s.kv_lora) : 0, 0);
    c.scales.assign(fmt == k::kDsaLatInt8 ? (size_t) (max_cells * s.kv_lora / k::DSA_LAT_GROUP) : 0, 0);
    c.kg.assign((size_t) (max_cells * 2 * s.idx_dim), 0);
    c.pooled.assign((size_t) (k::dsa_max_pools(max_cells, s) * s.idx_dim), 0);
}

void cache_append(HostCache& c, const DsaShapes& s, int64_t p0, int64_t n, const float* latent, const float* key,
                  const float* gate) {
    const int L = (int) s.kv_lora, G = L / k::DSA_LAT_GROUP, I = (int) s.idx_dim;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t cell = p0 + i;
        const float* x = latent + i * L;
        float* dst = c.latent.data() + cell * L;
        if (c.fmt == k::kDsaLatInt8) {
            int8_t* q = c.codes.data() + cell * L;
            uint16_t* sc = c.scales.data() + cell * G;
            quantize_latent_row(x, L, q, sc);
            for (int d = 0; d < L; ++d) dst[d] = (float) q[d] * k::f32_from_f16(sc[d / k::DSA_LAT_GROUP]);
        } else {
            for (int d = 0; d < L; ++d) dst[d] = f16_round(x[d]);   // set_rows f32 -> f16
        }
        for (int d = 0; d < I; ++d) {
            c.kg[(size_t) cell * 2 * I + d] = ggml_fp32_to_fp16(key[i * I + d]);
            c.kg[(size_t) cell * 2 * I + I + d] = ggml_fp32_to_fp16(gate[i * I + d]);
        }
        c.n_cells = std::max(c.n_cells, cell + 1);
    }
}

void oracle_pool(HostCache& c, const DsaShapes& s, const std::vector<float>& ape, int64_t j0, int64_t j1,
                 int n_threads) {
    const int64_t I = s.idx_dim, P = s.kpool;
    for (int64_t a = j0; a < j1; a += 4096) {   // bounded graphs
        const int64_t b = std::min(j1, a + 4096), n = b - a;
        Ctx wc(8 * ggml_tensor_overhead(), true);
        ggml_tensor* kg = wrap(wc, GGML_TYPE_F16, 2 * I, c.n_cells, 1, c.kg.data());
        ggml_tensor* ape_t = wrap(wc, GGML_TYPE_F32, I, P, 1, ape.data());
        Ctx cc((size_t) (n * P * 2 * I * 4) * 8 + (64 << 20), false);
        ggml_tensor* idx = ggml_new_tensor_1d(cc, GGML_TYPE_I32, P * n);
        int32_t* id = (int32_t*) idx->data;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t m = 0; m < P; ++m) id[j * P + m] = (int32_t) ((a + j) * P + m);
        // G:801-818, in order
        ggml_tensor* rows = ggml_get_rows(cc, kg, idx);
        rows = ggml_reshape_3d(cc, rows, 2 * I, P, n);
        ggml_tensor* pk = ggml_view_3d(cc, rows, I, P, n, rows->nb[1], rows->nb[2], 0);
        ggml_tensor* pg = ggml_view_3d(cc, rows, I, P, n, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, I));
        ggml_tensor* logits = ggml_add(cc, pg, ape_t);
        logits = ggml_cont(cc, ggml_permute(cc, logits, 1, 0, 2, 3));
        ggml_tensor* probs = ggml_soft_max(cc, ggml_reshape_2d(cc, logits, P, I * n));
        probs = ggml_reshape_3d(cc, probs, P, I, n);
        pk = ggml_cont(cc, ggml_permute(cc, pk, 1, 0, 2, 3));
        ggml_tensor* pooled = ggml_sum_rows(cc, ggml_mul(cc, probs, pk));
        pooled = ggml_reshape_2d(cc, pooled, I, n);
        compute(cc, {pooled}, n_threads);
        const float* pv = (const float*) pooled->data;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t d = 0; d < I; ++d) c.pooled[(size_t) (a + j) * I + d] = ggml_fp32_to_fp16(pv[j * I + d]);
    }
}

void oracle_project(const HostWeights& w, const DsaShapes& s, const std::vector<float>& x, int64_t T, StepOut& r,
                    int n_threads) {
    Ctx wc(32 * ggml_tensor_overhead(), true);
    const ggml_type Q = GGML_TYPE_Q8_0;
    ggml_tensor* wq_a = wrap(wc, Q, s.n_embd, s.q_lora, 1, w.wq_a.data());
    ggml_tensor* wq_b = wrap(wc, Q, s.q_lora, s.n_head * s.head_k, 1, w.wq_b.data());
    ggml_tensor* wkv_a = wrap(wc, Q, s.n_embd, s.kv_lora, 1, w.wkv_a.data());
    ggml_tensor* wk_b = wrap(wc, Q, s.head_k, s.kv_lora, s.n_head, w.wk_b.data());
    ggml_tensor* idx_q_b = wrap(wc, Q, s.q_lora, s.idx_n_head * s.idx_dim, 1, w.idx_q_b.data());
    ggml_tensor* idx_k = wrap(wc, Q, s.n_embd, s.idx_dim, 1, w.idx_k.data());
    ggml_tensor* idx_gate = wrap(wc, Q, s.n_embd, s.idx_dim, 1, w.idx_gate.data());
    ggml_tensor* idx_proj = wrap(wc, GGML_TYPE_F32, s.n_embd, s.idx_n_head, 1, w.idx_proj.data());
    ggml_tensor* q_a_norm = wrap(wc, GGML_TYPE_F32, s.q_lora, 1, 1, w.q_a_norm.data());
    ggml_tensor* kv_a_norm = wrap(wc, GGML_TYPE_F32, s.kv_lora, 1, 1, w.kv_a_norm.data());
    ggml_tensor* k_norm = wrap(wc, GGML_TYPE_F32, s.idx_dim, 1, 1, w.idx_k_norm.data());
    ggml_tensor* k_norm_b = wrap(wc, GGML_TYPE_F32, s.idx_dim, 1, 1, w.idx_k_norm_b.data());

    Ctx cc((size_t) T * (s.n_head * (s.head_k + s.kv_lora) * 3 + s.q_lora * 4 + s.n_embd * 2) * 4 + (64 << 20), false);
    ggml_tensor* cur = input_f32(cc, x.data(), s.n_embd, T);
    // build_dsa_layer (G:937-955)
    ggml_tensor* qa = ggml_mul_mat(cc, wq_a, cur);
    ggml_tensor* qr = ggml_mul(cc, ggml_rms_norm(cc, qa, s.rms_eps), q_a_norm);
    ggml_tensor* q = ggml_mul_mat(cc, wq_b, qr);
    q = ggml_reshape_3d(cc, q, s.head_k, s.n_head, T);
    ggml_tensor* kv_raw = ggml_mul_mat(cc, wkv_a, cur);
    ggml_tensor* kv = ggml_mul(cc, ggml_rms_norm(cc, kv_raw, s.rms_eps), kv_a_norm);
    ggml_tensor* q_abs = ggml_permute(cc, q, 0, 2, 1, 3);
    q_abs = ggml_mul_mat(cc, wk_b, q_abs);
    q_abs = ggml_permute(cc, q_abs, 0, 2, 1, 3);
    q_abs = ggml_cont(cc, q_abs);   // [kv_lora, n_head, T] for the readout
    // build_kpool_select (G:779-790, 832-834)
    ggml_tensor* iq = ggml_mul_mat(cc, idx_q_b, qr);
    ggml_tensor* ik_raw = ggml_mul_mat(cc, idx_k, cur);
    ggml_tensor* ik = ggml_add(cc, ggml_mul(cc, ggml_norm(cc, ik_raw, s.norm_eps), k_norm), k_norm_b);
    ggml_tensor* ig = ggml_mul_mat(cc, idx_gate, cur);
    ggml_tensor* iw = ggml_mul_mat(cc, idx_proj, cur);
    iw = ggml_scale(cc, iw, 1.0f / sqrtf(float(s.idx_dim * s.idx_n_head)));
    ggml_tensor* qc = ggml_cont(cc, q);
    compute(cc, {qr, qc, kv, q_abs, iq, ik, ig, iw}, n_threads);
    read(r.qa, qa);
    read(r.kv, kv_raw);
    read(r.ik_raw, ik_raw);
    read(r.qr, qr);
    read(r.q, qc);
    read(r.latent, kv);
    read(r.q_abs, q_abs);
    read(r.iq, iq);
    read(r.ik, ik);
    read(r.ig, ig);
    read(r.iw, iw);
}

void oracle_select(const HostCache& c, const DsaShapes& s, const StepOut& a, int64_t p0, int64_t T, StepOut& r,
                   int n_threads) {
    const int64_t I = s.idx_dim, H = s.idx_n_head, P = s.kpool;
    const int64_t n_real = (p0 + T) / P;   // pools complete at the window's last token
    const int64_t n_pool = kpool_pad((uint32_t) n_real);
    const int64_t n_top = std::min<int64_t>(n_pool, s.idx_top_k / P);
    Ctx wc(8 * ggml_tensor_overhead(), true);
    ggml_tensor* pooled_all = wrap(wc, GGML_TYPE_F16, I, (int64_t) (c.pooled.size() / I), 1, c.pooled.data());
    Ctx cc((size_t) (n_pool * (I + T * 8) * 4 * 4) + (64 << 20), false);
    ggml_tensor* pool_cells = ggml_new_tensor_1d(cc, GGML_TYPE_I32, n_pool);
    for (int64_t j = 0; j < n_pool; ++j) ((int32_t*) pool_cells->data)[j] = (int32_t) (j < n_real ? j : 0);
    ggml_tensor* mask = ggml_new_tensor_2d(cc, GGML_TYPE_F16, n_pool, T);
    for (int64_t t = 0; t < T; ++t) {
        const int64_t nv = (p0 + t + 1) / P;   // pools ending at or before the token
        for (int64_t j = 0; j < n_pool; ++j)
            ((ggml_fp16_t*) mask->data)[t * n_pool + j] = ggml_fp32_to_fp16(j < nv ? 0.0f : -INFINITY);
    }
    ggml_tensor* iq = input_f32(cc, a.iq.data(), I, H, T);
    ggml_tensor* weights = input_f32(cc, a.iw.data(), H, T);
    // G:827-871
    ggml_tensor* pooled = ggml_get_rows(cc, pooled_all, pool_cells);
    pooled = ggml_reshape_3d(cc, pooled, I, 1, n_pool);
    ggml_tensor* score = ggml_lightning_indexer(cc, iq, pooled, weights, mask);
    ggml_tensor* top_k = ggml_top_k(cc, score, (int) n_top);
    ggml_tensor* sel_score = ggml_get_rows(cc, ggml_reshape_3d(cc, score, 1, n_pool, T), top_k);
    ggml_tensor* sel_order = ggml_argsort(cc, ggml_reshape_2d(cc, sel_score, n_top, T), GGML_SORT_ORDER_DESC);
    top_k = ggml_get_rows(cc, ggml_reshape_3d(cc, ggml_cast(cc, top_k, GGML_TYPE_F32), 1, n_top, T), sel_order);
    top_k = ggml_cast(cc, ggml_cont(cc, ggml_reshape_2d(cc, top_k, n_top, T)), GGML_TYPE_I32);
    compute(cc, {score, top_k}, n_threads);
    r.n_pool = n_pool;
    read(r.scores, score);
    // the pool -> cell expansion and the tail (set_input_kpool's pool_idxs, tail_idxs, gather_mask)
    const int32_t* tk = (const int32_t*) top_k->data;
    r.sel.assign((size_t) T, {});
    r.slots.assign((size_t) T, {});
    for (int64_t t = 0; t < T; ++t) {
        const int64_t pos = p0 + t, nv = (pos + 1) / P, nvc = std::min(nv, n_top), n_tail = (pos + 1) % P;
        auto& slots = r.slots[(size_t) t];
        for (int64_t sl = 0; sl < n_top; ++sl) {
            const int32_t pool = tk[t * n_top + sl];
            for (int64_t m = 0; m < P; ++m) slots.push_back(sl < nvc ? (int32_t) (pool * P + m) : -1);
        }
        for (int64_t m = 0; m < P - 1; ++m) slots.push_back(m < n_tail ? (int32_t) (pos - m) : -1);
        auto& sel = r.sel[(size_t) t];
        for (int32_t cell : slots)
            if (cell >= 0) sel.push_back(cell);
        std::sort(sel.begin(), sel.end());
    }
}

void oracle_attend(const HostWeights& w, const HostCache& c, const DsaShapes& s, const StepOut& a, int64_t p0,
                   int64_t T, const std::vector<std::vector<int32_t>>& slots, StepOut& r, int n_threads) {
    const int64_t L = s.kv_lora, H = s.n_head;
    int64_t n_sel = 0;
    for (const auto& v : slots) n_sel = std::max<int64_t>(n_sel, (int64_t) v.size());
    Ctx wc(8 * ggml_tensor_overhead(), true);
    ggml_tensor* wv_b = wrap(wc, GGML_TYPE_Q8_0, L, s.head_v, H, w.wv_b.data());
    ggml_tensor* wo = wrap(wc, GGML_TYPE_Q8_0, H * s.head_v, s.n_embd, 1, w.wo.data());
    std::vector<uint16_t> lat16;
    ggml_tensor* lat;
    if (c.fmt == k::kDsaLatF16) {
        lat16.resize((size_t) (c.n_cells * L));
        for (size_t i = 0; i < lat16.size(); ++i) lat16[i] = ggml_fp32_to_fp16(c.latent[i]);   // exact: f16 values
        lat = wrap(wc, GGML_TYPE_F16, L, c.n_cells, 1, lat16.data());
    } else {
        lat = wrap(wc, GGML_TYPE_F32, L, c.n_cells, 1, c.latent.data());   // the int8 cache, dequantized
    }
    Ctx cc((size_t) (n_sel * T * L * 4) * 4 + (size_t) (T * H * (L + s.head_v) * 4) * 4 + (128 << 20), false);
    ggml_tensor* sel_idx = ggml_new_tensor_1d(cc, GGML_TYPE_I32, n_sel * T);
    ggml_tensor* gmask = ggml_new_tensor_4d(cc, GGML_TYPE_F32, n_sel, 1, 1, T);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < n_sel; ++j) {
            const int32_t cell = j < (int64_t) slots[(size_t) t].size() ? slots[(size_t) t][(size_t) j] : -1;
            ((int32_t*) sel_idx->data)[t * n_sel + j] = cell >= 0 ? cell : (int32_t) p0;   // gather's dummy cell
            ((float*) gmask->data)[t * n_sel + j] = cell >= 0 ? 0.0f : -INFINITY;
        }
    ggml_tensor* q_abs = input_f32(cc, a.q_abs.data(), L, H, T);
    // G:974-990
    ggml_tensor* k_g = ggml_get_rows(cc, lat, sel_idx);
    k_g = ggml_reshape_4d(cc, k_g, L, n_sel, 1, T);
    ggml_tensor* q_g = ggml_permute(cc, q_abs, 0, 2, 3, 1);
    ggml_tensor* kq = ggml_mul_mat(cc, k_g, q_g);
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(cc, kq, gmask, 1.0f / sqrtf((float) s.head_k), 0.0f);
    ggml_tensor* v_t = ggml_cont(cc, ggml_transpose(cc, k_g));
    ggml_tensor* kqv = ggml_mul_mat(cc, v_t, kq);
    ggml_tensor* o = ggml_mul_mat(cc, wv_b, kqv);
    ggml_tensor* out = ggml_cont(cc, ggml_permute(cc, o, 0, 2, 1, 3));
    out = ggml_reshape_2d(cc, out, s.head_v * H, T);
    out = ggml_mul_mat(cc, wo, out);
    compute(cc, {kqv, o, out}, n_threads);
    read(r.o_lat, kqv);
    read(r.o, o);
    read(r.out, out);
}

Err compare(const float* got, const float* ref, int64_t n) {
    Err e;
    e.n = n;
    for (int64_t i = 0; i < n; ++i) {
        e.max_abs = std::max(e.max_abs, (double) std::fabs((double) got[i] - (double) ref[i]));
        e.ref_max = std::max(e.ref_max, (double) std::fabs((double) ref[i]));
    }
    e.rel = e.ref_max > 0 ? e.max_abs / e.ref_max : e.max_abs;
    return e;
}

FlipBound q8_1_flip_bound(const float* x, const void* gpu_q8_1, int64_t cols, int64_t n_in) {
    FlipBound fb;
    const uint8_t* gq = static_cast<const uint8_t*>(gpu_q8_1);
    const int64_t nb = n_in / 32;
    for (int64_t c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (int64_t b = 0; b < nb; ++b) {
            const float* v = x + c * n_in + b * 32;
            const uint8_t* blk = gq + (size_t) (c * nb + b) * 36;   // Q8_1: half2 {d, sum}, then 32 int8 codes
            uint16_t dg_bits;
            std::memcpy(&dg_bits, blk, 2);
            const int8_t* qg = reinterpret_cast<const int8_t*>(blk + 4);
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(v[i]));
            if (amax == 0.0f) {   // both sides: d = 0, every code 0
                for (int i = 0; i < 32; ++i) fb.bad += qg[i] != 0;
                fb.bad += (dg_bits & 0x7fff) != 0;
                continue;
            }
            const float d = amax / 127.0f, id = 127.0f / amax;
            const uint16_t dc_bits = k::f16_from_f32(d);   // ggml's Q8_0 d, GGML_CPU_FP32_TO_FP16
            // the GPU's d is amax / 127 too, divided approximately under CUDA's fast math: one f16 ulp at most
            if (std::abs((int) dg_bits - (int) dc_bits) > 1 || ((dg_bits ^ dc_bits) & 0x8000)) ++fb.bad;
            const double dg = k::f32_from_f16(dg_bits), dc = k::f32_from_f16(dc_bits);
            for (int i = 0; i < 32; ++i) {
                float r = roundf(v[i] / d);   // the GPU's rule, with IEEE division
                r = r > 127.0f ? 127.0f : (r < -127.0f ? -127.0f : r);
                const float cpu = std::nearbyint(v[i] * id);
                if ((float) qg[i] != r) {
                    // a near-tie only: the exact quotient within 2^-20 of the half-integer between the two codes
                    // (CUDA's div.approx.f32 is within 2 ulp, twice: d, then v / d)
                    const double q = (double) v[i] / (double) d, h = 0.5 * ((double) qg[i] + (double) r);
                    if (std::fabs((double) qg[i] - (double) r) == 1.0 && std::fabs(q - h) <= std::fabs(h) * 0x1p-20)
                        ++fb.off_rule;
                    else
                        ++fb.bad;
                }
                const double diff = std::fabs(dg * (double) qg[i] - dc * (double) cpu);
                if (diff > 0.0) {
                    ++fb.flips;
                    sum += diff;
                }
            }
        }
        fb.steps = std::max(fb.steps, sum);
    }
    return fb;
}

double q8_0_max_abs(const std::vector<uint8_t>& w) {
    double m = 0.0;
    for (size_t b = 0; b + 34 <= w.size(); b += 34) {
        uint16_t dbits;
        std::memcpy(&dbits, w.data() + b, 2);
        const double d = std::fabs((double) ggml_fp16_to_fp32(dbits));
        int qmax = 0;
        for (int i = 0; i < 32; ++i) qmax = std::max(qmax, std::abs((int) (int8_t) w[b + 2 + i]));
        m = std::max(m, d * qmax);
    }
    return m;
}

}  // namespace dsa_test
