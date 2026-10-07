// src/core/glm_seldump.cpp - see include/strata/core/glm_seldump.hpp.
#include "strata/core/glm_seldump.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace strata::core {

const GlmSelDump& GlmSelDump::get() {
    static const GlmSelDump d = [] {
        GlmSelDump r;
        const char* dir = std::getenv("STRATA_GLM_SEL_DUMP");
        const char* at = std::getenv("STRATA_GLM_SEL_AT");
        if (dir == nullptr || *dir == '\0') return r;
        r.dir = dir;
        if (at != nullptr) {
            std::stringstream ss(at);
            std::string item;
            while (std::getline(ss, item, ','))
                if (!item.empty()) r.at.push_back(std::atoll(item.c_str()));
        }
        std::sort(r.at.begin(), r.at.end());
        r.at.erase(std::unique(r.at.begin(), r.at.end()), r.at.end());
        r.on = !r.at.empty();
        if (!r.on) std::fprintf(stderr, "strata: STRATA_GLM_SEL_DUMP is set without STRATA_GLM_SEL_AT: no selection dump\n");
        return r;
    }();
    return d;
}

bool GlmSelDump::has(int64_t pos) const { return on && std::binary_search(at.begin(), at.end(), pos); }

bool GlmSelDump::any_in(int64_t p0, int64_t n) const {
    if (!on) return false;
    auto it = std::lower_bound(at.begin(), at.end(), p0);
    return it != at.end() && *it < p0 + n;
}

bool GlmSelDump::write(int64_t layer, int64_t pos, const int32_t* ids, const float* scores,
                       const kernels::DsaShapes& s, void* stream, const char* src, std::string& err) const {
    const int64_t n_bid = (pos + 1) / s.kpool, width = kernels::dsa_sel_width(pos, s);
    std::vector<int32_t> h_ids((size_t) width);
    std::vector<float> h_sc(scores != nullptr ? (size_t) n_bid : 0);
    cudaStream_t st = (cudaStream_t) stream;
    if (ids != nullptr) cudaMemcpyAsync(h_ids.data(), ids, (size_t) width * 4, cudaMemcpyDeviceToHost, st);
    if (!h_sc.empty()) cudaMemcpyAsync(h_sc.data(), scores, h_sc.size() * 4, cudaMemcpyDeviceToHost, st);
    if (const cudaError_t e = cudaStreamSynchronize(st); e != cudaSuccess) {
        err = std::string("glm: selection dump: ") + cudaGetErrorString(e);
        return false;
    }
    if (ids == nullptr)
        for (int64_t i = 0; i < width; ++i) h_ids[(size_t) i] = (int32_t) i;
    const std::string path = dir + "/sel-L" + std::to_string(layer) + "-P" + std::to_string(pos) + ".txt";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        err = "glm: cannot write " + path;
        return false;
    }
    std::fprintf(f, "glm-sel 1\nsrc strata-%s\nlayer %lld\npos %lld\nn_bid %lld\nwidth %lld\nids", src, (long long) layer,
                 (long long) pos, (long long) n_bid, (long long) width);
    for (const int32_t v : h_ids) std::fprintf(f, " %d", v);
    std::fprintf(f, "\n");
    if (!h_sc.empty()) {
        std::fprintf(f, "scores");
        for (const float v : h_sc) std::fprintf(f, " %.9g", (double) v);
        std::fprintf(f, "\n");
    }
    if (std::fclose(f) != 0) {
        err = "glm: writing " + path;
        return false;
    }
    return true;
}

}  // namespace strata::core
