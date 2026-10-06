// src/kernels/cuda/mla_kv.cu - the DSA layer's latent KV cache: layout, append, page read, streaming view.
// See include/strata/kernels/dsa.hpp.
#include "strata/kernels/dsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mla_kv: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void validate(const DsaShapes& s, const char* what) {
    if (s.kv_lora <= 0 || s.kv_lora % DSA_LAT_GROUP != 0 || s.idx_dim <= 0 || s.idx_dim % 2 != 0 || s.kpool < 2 ||
        s.idx_top_k % s.kpool != 0) {
        std::fprintf(stderr, "mla_kv: %s: unsupported geometry (kv_lora %lld must be a multiple of %d, kpool %lld)\n",
                     what, (long long) s.kv_lora, DSA_LAT_GROUP, (long long) s.kpool);
        std::exit(1);
    }
}

uint64_t align256(uint64_t n) { return (n + 255) & ~255ull; }

struct Cursor {
    uint8_t* p;
    uint64_t used = 0;
    template <typename T>
    T* take(uint64_t count) {
        T* r = p ? (T*) (p + used) : nullptr;
        used = align256(used + count * sizeof(T));
        return r;
    }
};

// The per-slot arrays of a page store, in the order both the VRAM arena and the host copy lay them out.
struct PageArrays {
    int8_t* lat_q = nullptr;
    uint16_t* lat_scale = nullptr;
    uint16_t* lat_f16 = nullptr;
    uint16_t* kg = nullptr;
    uint16_t* pad = nullptr;
};
PageArrays carve_pages(Cursor& c, const DsaShapes& s, int fmt, int64_t pages) {
    PageArrays a;
    const uint64_t cells = (uint64_t) pages * s.kpool;
    if (fmt == kDsaLatInt8) {
        a.lat_q = c.take<int8_t>(cells * s.kv_lora);
        a.lat_scale = c.take<uint16_t>(cells * (s.kv_lora / DSA_LAT_GROUP));
        a.kg = c.take<uint16_t>(cells * 2 * s.idx_dim);
        a.pad = c.take<uint16_t>(cells * (s.kv_lora / DSA_LAT_GROUP));
    } else {
        a.lat_f16 = c.take<uint16_t>(cells * s.kv_lora);
        a.kg = c.take<uint16_t>(cells * 2 * s.idx_dim);
    }
    return a;
}

// One block per cell: the latent's groups (int8: one f16 scale per 64 values, kv_q8.hpp's rounding) and the
// indexer key | gate rows as f16. `pos_of` gives the cell's position: from the step records, or p0 + i.
struct AppendArgs {
    DsaKvCache c;
    const float* latent;
    const float* key;
    const float* gate;
    const int32_t* steps;
    long long p0;
    int kv_lora, idx_dim, kpool;
};

__global__ void __launch_bounds__(256) append_kernel(AppendArgs a) {
    const int i = blockIdx.x, t = threadIdx.x;
    const long long pos = a.steps ? (long long) a.steps[i * kStepCount + kStepPos] : a.p0 + i;
    const long long page = pos / a.kpool, in_page = pos % a.kpool;
    const int slot = a.c.page_table[page];
    const long long hrow = page * a.kpool + in_page;                     // identity layout of the host copy
    const long long vrow = slot >= 0 ? (long long) slot * a.kpool + in_page : -1;
    const float* x = a.latent + (size_t) i * a.kv_lora;
    const int groups = a.kv_lora / DSA_LAT_GROUP;
    __shared__ float s_scale[64];
    __shared__ uint16_t s_bits[64];
    if (a.c.fmt == kDsaLatInt8) {
        // max |x| per group of 64: one thread per group, sequentially (exact: max is order-free)
        for (int g = t; g < groups; g += blockDim.x) {
            float amax = 0.0f;
            for (int j = 0; j < DSA_LAT_GROUP; ++j) amax = fmaxf(amax, fabsf(x[g * DSA_LAT_GROUP + j]));
            const uint16_t sbits = f16_from_f32(amax / 127.0f);
            s_bits[g] = sbits;
            s_scale[g] = f32_from_f16(sbits);                        // quantize against the STORED scale
        }
        __syncthreads();
        for (int d = t; d < a.kv_lora; d += blockDim.x) {
            const float sf = s_scale[d / DSA_LAT_GROUP];
            int q = 0;
            if (sf > 0.0f) {
                q = __float2int_rn(x[d] / sf);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
            }
            if (vrow >= 0) a.c.lat_q[vrow * a.kv_lora + d] = (int8_t) q;
            if (a.c.host.lat_q) a.c.host.lat_q[hrow * a.kv_lora + d] = (int8_t) q;
        }
        for (int g = t; g < groups; g += blockDim.x) {
            if (vrow >= 0) a.c.lat_scale[vrow * groups + g] = s_bits[g];
            if (a.c.host.lat_scale) a.c.host.lat_scale[hrow * groups + g] = s_bits[g];
        }
    } else {
        for (int d = t; d < a.kv_lora; d += blockDim.x) {
            const uint16_t h = f16_from_f32(x[d]);
            if (vrow >= 0) a.c.lat_f16[vrow * a.kv_lora + d] = h;
            if (a.c.host.lat_f16) a.c.host.lat_f16[hrow * a.kv_lora + d] = h;
        }
    }
    const int w = 2 * a.idx_dim;
    for (int d = t; d < w; d += blockDim.x) {
        const float v = d < a.idx_dim ? a.key[(size_t) i * a.idx_dim + d] : a.gate[(size_t) i * a.idx_dim + d - a.idx_dim];
        const uint16_t h = f16_from_f32(v);
        if (vrow >= 0) a.c.kg[vrow * w + d] = h;
        if (a.c.host.kg) a.c.host.kg[hrow * w + d] = h;
    }
}

__global__ void read_page_kernel(DsaKvCache c, long long page, int kv_lora, int idx_dim, int kpool, float* latent,
                                 float* kg) {
    const int slot = c.page_table[page];
    const bool res = slot >= 0;
    const long long base = res ? (long long) slot * kpool : page * kpool;
    const int groups = kv_lora / DSA_LAT_GROUP;
    for (int i = threadIdx.x; i < kpool * kv_lora; i += blockDim.x) {
        const long long row = base + i / kv_lora;
        const int d = i % kv_lora;
        float v;
        if (c.fmt == kDsaLatInt8) {
            const int8_t* q = res ? c.lat_q : c.host.lat_q;
            const uint16_t* sc = res ? c.lat_scale : c.host.lat_scale;
            v = (float) q[row * kv_lora + d] * f32_from_f16(sc[row * groups + d / DSA_LAT_GROUP]);
        } else {
            v = f32_from_f16((res ? c.lat_f16 : c.host.lat_f16)[row * kv_lora + d]);
        }
        if (latent) latent[i] = v;
    }
    const int w = 2 * idx_dim;
    for (int i = threadIdx.x; i < kpool * w; i += blockDim.x) {
        const long long row = base + i / w;
        if (kg) kg[i] = f32_from_f16((res ? c.kg : c.host.kg)[row * w + i % w]);
    }
}

__global__ void reset_table_kernel(int32_t* table, long long n, bool identity) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
        table[i] = identity ? (int32_t) i : -1;
}

}  // namespace

void dsa_step_fill(int32_t* host_steps, int64_t p0, int64_t T, const DsaShapes& s) {
    for (int64_t i = 0; i < T; ++i) {
        const int64_t pos = p0 + i;
        int32_t* st = host_steps + i * kStepCount;
        st[kStepPos] = (int32_t) pos;
        st[kStepNKv] = (int32_t) (pos + 1);
        st[kStepNBid] = (int32_t) ((pos + 1) / s.kpool);
        st[kStepWidth] = (int32_t) dsa_sel_width(pos, s);
    }
}

uint64_t dsa_kv_page_bytes(const DsaShapes& s, int fmt) {
    const uint64_t kg = (uint64_t) s.kpool * 2 * s.idx_dim * 2;
    if (fmt == kDsaLatInt8)
        return (uint64_t) s.kpool * s.kv_lora + (uint64_t) s.kpool * (s.kv_lora / DSA_LAT_GROUP) * 2 * 2 + kg;
    return (uint64_t) s.kpool * s.kv_lora * 2 + kg;
}

uint64_t dsa_kv_cell_bytes(const DsaShapes& s, int fmt) {
    return dsa_kv_page_bytes(s, fmt) / s.kpool + (uint64_t) s.idx_dim * 2 / s.kpool;
}

uint64_t dsa_kv_carve(void* base, const DsaShapes& s, int fmt, int64_t max_cells, int64_t n_slots, DsaKvCache& c) {
    validate(s, "carve");
    const int64_t pages = (max_cells + s.kpool - 1) / s.kpool;
    if (n_slots <= 0 || n_slots > pages) n_slots = pages;
    Cursor cur{(uint8_t*) base};
    const PageArrays a = carve_pages(cur, s, fmt, n_slots);
    c.fmt = fmt;
    c.lat_q = a.lat_q;
    c.lat_scale = a.lat_scale;
    c.lat_f16 = a.lat_f16;
    c.kg = a.kg;
    c.pad = a.pad;
    c.page_table = cur.take<int32_t>((uint64_t) pages);
    c.max_pools = dsa_max_pools(max_cells, s);
    c.pooled = cur.take<uint16_t>((uint64_t) c.max_pools * s.idx_dim);
    c.n_pages = pages;
    c.n_slots = n_slots;
    c.max_cells = max_cells;
    c.map = KvStreamMap{};
    c.map.page_table = c.page_table;
    c.map.n_blocks = pages;
    c.map.n_slots = n_slots;
    if (n_slots < pages) {
        c.map.slot_block = cur.take<int32_t>((uint64_t) n_slots);
        c.map.slot_stamp = cur.take<int32_t>((uint64_t) n_slots);
        c.map.slot_ref = cur.take<int32_t>((uint64_t) n_slots);
        c.map.miss_block = cur.take<int32_t>((uint64_t) n_slots);
        c.map.miss_slot = cur.take<int32_t>((uint64_t) n_slots);
        c.map.ctl = cur.take<int32_t>((uint64_t) kKvCtlInts);
    }
    return cur.used;
}

uint64_t dsa_kv_bytes(const DsaShapes& s, int fmt, int64_t max_cells, int64_t n_slots) {
    DsaKvCache c;
    return dsa_kv_carve(nullptr, s, fmt, max_cells, n_slots, c) + 256;
}

uint64_t dsa_kv_host_bytes(const DsaShapes& s, int fmt, int64_t max_cells) {
    validate(s, "host bytes");
    Cursor cur{nullptr};
    carve_pages(cur, s, fmt, (max_cells + s.kpool - 1) / s.kpool);
    return cur.used + 256;
}

void dsa_kv_host_carve(void* dev_base, const DsaShapes& s, int fmt, int64_t max_cells, DsaKvCache& c) {
    validate(s, "host carve");
    Cursor cur{(uint8_t*) dev_base};
    const PageArrays a = carve_pages(cur, s, fmt, (max_cells + s.kpool - 1) / s.kpool);
    c.host.lat_q = a.lat_q;
    c.host.lat_scale = a.lat_scale;
    c.host.lat_f16 = a.lat_f16;
    c.host.kg = a.kg;
    c.host.pad = a.pad;
}

void dsa_kv_reset(DsaKvCache& c, const DsaShapes& s, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    const uint64_t slot_cells = (uint64_t) c.n_slots * s.kpool;
    if (c.lat_q) (void) cudaMemsetAsync(c.lat_q, 0, slot_cells * s.kv_lora, st);
    if (c.lat_scale) (void) cudaMemsetAsync(c.lat_scale, 0, slot_cells * (s.kv_lora / DSA_LAT_GROUP) * 2, st);
    if (c.lat_f16) (void) cudaMemsetAsync(c.lat_f16, 0, slot_cells * s.kv_lora * 2, st);
    if (c.kg) (void) cudaMemsetAsync(c.kg, 0, slot_cells * 2 * s.idx_dim * 2, st);
    if (c.pad) (void) cudaMemsetAsync(c.pad, 0, slot_cells * (s.kv_lora / DSA_LAT_GROUP) * 2, st);
    (void) cudaMemsetAsync(c.pooled, 0, (size_t) c.max_pools * s.idx_dim * 2, st);
    const bool streamed = c.n_slots < c.n_pages;
    reset_table_kernel<<<64, 256, 0, st>>>(c.page_table, c.n_pages, !streamed);
    check("reset");
    if (streamed) kv_stream_reset(c.map, stream);
}

void dsa_kv_append(const DsaKvCache& c, const DsaShapes& s, const int32_t* steps, int64_t T, const float* latent,
                   const float* key, const float* gate, void* stream) {
    validate(s, "append");
    if (T <= 0) return;
    if (s.kv_lora / DSA_LAT_GROUP > 64) {
        std::fprintf(stderr, "mla_kv: append: kv_lora %lld exceeds 64 groups\n", (long long) s.kv_lora);
        std::exit(1);
    }
    AppendArgs a{c, latent, key, gate, steps, 0, (int) s.kv_lora, (int) s.idx_dim, (int) s.kpool};
    append_kernel<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(a);
    check("append");
}

void dsa_kv_append_range(const DsaKvCache& c, const DsaShapes& s, int64_t p0, int64_t n, const float* latent,
                         const float* key, const float* gate, void* stream) {
    validate(s, "append range");
    if (n <= 0) return;
    AppendArgs a{c, latent, key, gate, nullptr, (long long) p0, (int) s.kv_lora, (int) s.idx_dim, (int) s.kpool};
    append_kernel<<<(unsigned) n, 256, 0, (cudaStream_t) stream>>>(a);
    check("append range");
}

void dsa_kv_read_page(const DsaKvCache& c, const DsaShapes& s, int64_t page, float* latent, float* kg, void* stream) {
    validate(s, "read page");
    if (page < 0 || page >= c.n_pages) {
        std::fprintf(stderr, "mla_kv: read page %lld outside [0, %lld)\n", (long long) page, (long long) c.n_pages);
        std::exit(1);
    }
    read_page_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(c, (long long) page, (int) s.kv_lora, (int) s.idx_dim,
                                                           (int) s.kpool, latent, kg);
    check("read page");
}

void dsa_kv_stream_resolve(const DsaKvCache& c, const DsaShapes& s, const int32_t* ids, const int32_t* steps,
                           int64_t T, int64_t cap, void* stream) {
    if (c.n_slots >= c.n_pages || !c.host.present()) return;   // fully resident
    // The int8 page as kv_stream.cu's int8 QSA page: one KV head of kv_lora values, K = the latent (codes, scales),
    // V codes = the key | gate rows, V scales = the pad. The V run is kpool * kv_lora bytes; the key | gate rows
    // are kpool * 2 * idx_dim f16 = kpool * 4 * idx_dim bytes: equal iff 4 * idx_dim == kv_lora (GLM: 512).
    if (c.fmt != kDsaLatInt8 || 4 * s.idx_dim != s.kv_lora) {
        std::fprintf(stderr, "mla_kv: streaming needs the int8 latent and 4 * idx_dim == kv_lora (%lld, %lld)\n",
                     (long long) s.idx_dim, (long long) s.kv_lora);
        std::exit(1);
    }
    QsaShapes q{};
    q.n_head = 1;
    q.n_head_kv = 1;
    q.head_dim = s.kv_lora;
    q.page_size = s.kpool;
    q.idx_block = s.kpool;
    QsaAttnPools slots{};
    slots.k_q = c.lat_q;
    slots.k_scale = c.lat_scale;
    slots.v_q = (const int8_t*) c.kg;
    slots.v_scale = c.pad;
    slots.page_table = c.page_table;
    KvHostPools host{};
    host.k_q = c.host.lat_q;
    host.k_scale = c.host.lat_scale;
    host.v_q = (int8_t*) c.host.kg;
    host.v_scale = c.host.pad;
    static_assert(KV_Q8_GROUP == DSA_LAT_GROUP, "the latent's scale group must be kv_q8's");
    kv_stream_resolve(c.map, slots, host, kKvInt8, ids, steps, T, cap, q, stream);
}

}  // namespace strata::kernels
