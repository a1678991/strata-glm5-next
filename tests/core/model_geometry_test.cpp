// tests/core/model_geometry_test.cpp - the architecture guard, ModelGeometry from a model file, the tensor-directory
// check and the native loaders' acceptance, for both architectures, on header-only GGUFs of the real files (no
// weights, no GPU call).
//
// The fixtures are written by tests/core/geometry_fixtures.py from tests/data/gguf_headers/*.json (the metadata and
// tensor directories of GLM-5.3-Flash UD-IQ4_XS and Qwen3.8-Flash-Next UD-IQ4_XS, read on 2026-10-06), plus variants
// with one change each that must be refused.  Usage: model_geometry_test FIXTURE_DIR
//
//   1. qwen4exp: the guard as before; the geometry is the compiled-in one, field for field (the 18 fields of the
//      conversation snapshot's geometry key and the layer predicate), with the two keys read from the file.
//   2. glm5-next: the guard; every geometry field against the plan's table (docs/strata-glm-port.md 1.2 and 2.2 in
//      the deployment repo); the layer kinds from attention.head_count_kv; check_all_gguf covers EVERY tensor of the
//      file (1,412 = what the checks compared).
//   3. Refusals: a 2-latent MLA layer, a pinned shape key, one tensor's shape, a layer whose kind and tensors
//      disagree, a per-layer clamp that differs, an unknown architecture.
//   4. The native loaders on both files: NativeHead (Q6_K [n_embd, vocab]) and NativeEmbed (Q8_0) pass the
//      architecture, type and shape checks and stop at the (absent) payload; NativeDense serves GLM's dense FFN.
//   5. Native expert formats at GLM's widths (4096 / 2048): blob sizes of the three format pairs of the file and a
//      45-layer native_experts.txt with three dense layers (STRATA_NATIVE_EXPERTS builds).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::LayerKind;
using strata::core::ModelGeometry;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

/// The shards of the fixture in `dir`, metadata shard first (any shard names the family).
std::vector<std::string> shards_of(const fs::path& dir) {
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".gguf") return strata::gguf_split_paths(e.path().string());
    return {};
}

/// geometry_from_gguf on the fixture in `dir`; `err` gets the refusal or an exception.
bool geometry(const fs::path& dir, ModelGeometry& g, std::string& err) {
    err.clear();
    try {
        const strata::GgufModel m(shards_of(dir));
        return strata::core::geometry_from_gguf(m.meta(), g, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool tensors_ok(const fs::path& dir, const ModelGeometry& g, std::string& err, int64_t* checked = nullptr) {
    err.clear();
    try {
        const strata::GgufModel m(shards_of(dir));
        return strata::core::check_all_gguf(m, g, err, checked);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

size_t file_tensors(const fs::path& dir) {
    const strata::GgufModel m(shards_of(dir));
    size_t n = 0;
    for (size_t i = 0; i < m.size(); ++i) n += m.shard(i).tensors().size();
    return n;
}

void qwen(const fs::path& root) {
    std::printf("qwen4exp\n");
    const fs::path dir = root / "qwen";
    {
        const strata::GgufModel m(shards_of(dir));
        strata::ModelArch a = strata::ModelArch::Unknown;
        check(strata::check_architecture(m.meta()).empty(), "check_architecture (the Qwen-only guard) accepts it");
        check(strata::check_model_architecture(m.meta(), &a).empty() && a == strata::ModelArch::Qwen4Exp,
              "check_model_architecture: qwen4exp");
        check(has(strata::check_glm5_next(m.meta()), "not 'glm5-next'"), "the glm5-next guard refuses it");
    }
    ModelGeometry g;
    std::string err;
    const bool read = geometry(dir, g, err);
    check(read, "geometry_from_gguf" + (err.empty() ? std::string() : ": " + err));
    // the compiled-in geometry, written out: these are the 18 fields of the conversation snapshot's geometry key
    const std::array<int64_t, 18> want = {2560, 48, 4, 128, 16, 48, 4, 10240, 6144, 24, 2, 256, 4, 128, 4, 320, 512, 640};
    const std::array<int64_t, 18> got = {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
                                         g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
                                         g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
                                         g.hc, g.hc_lr, g.n_expert, g.n_ff};
    check(got == want, "the 18 geometry-key fields are the compiled-in Qwen values");
    const ModelGeometry d;
    const std::array<int64_t, 18> dflt = {d.n_embd, d.n_layers, d.qsa_interval, d.ssm_state_size, d.ssm_k_heads,
                                          d.ssm_v_heads, d.ssm_d_conv, d.ssm_conv_channels, d.ssm_value_dim,
                                          d.n_head, d.n_head_kv, d.head_dim, d.idx_q_heads, d.idx_key_dim,
                                          d.hc, d.hc_lr, d.n_expert, d.n_ff};
    check(dflt == want, "a default-constructed ModelGeometry is the same (unchanged defaults)");
    check(!g.glm() && g.n_expert_used == 10 && g.n_vocab == 248320 && g.n_expert_shared == 1,
          "arch qwen4exp, top-10 from the file, vocabulary 248320 (the literal generate.cpp used)");
    bool pred = true;
    for (int64_t l = 0; l < 48; ++l) {
        pred = pred && strata::core::is_qsa_layer(g, l) == (l % 4 == 3);
        pred = pred && strata::core::layer_kind(g, l) == (l % 4 == 3 ? LayerKind::QsaMoe : LayerKind::GdnMoe);
    }
    check(pred && g.n_qsa_layers() == 12 && g.n_gdn_layers() == 36, "QSA at l % 4 == 3: 12 QSA, 36 GDN");
    const bool lists = tensors_ok(dir, g, err);
    check(lists, "the Qwen tensor lists match the real file's directory" + (err.empty() ? std::string() : ": " + err));
}

void glm(const fs::path& root) {
    std::printf("glm5-next\n");
    const fs::path dir = root / "glm";
    {
        const strata::GgufModel m(shards_of(dir));
        strata::ModelArch a = strata::ModelArch::Unknown;
        check(has(strata::check_architecture(m.meta()), "requires 'qwen4exp'"),
              "check_architecture (the Qwen-only guard, e.g. the SYCL engine's) still refuses it");
        check(strata::check_model_architecture(m.meta(), &a).empty() && a == strata::ModelArch::Glm5Next,
              "check_model_architecture: glm5-next");
    }
    ModelGeometry g;
    std::string err;
    const bool read = geometry(dir, g, err);
    check(read, "geometry_from_gguf" + (err.empty() ? std::string() : ": " + err));
    check(g.glm() && g.n_embd == 4096 && g.n_layers == 45 && g.n_layers_mtp == 1 && g.n_vocab == 154880,
          "n_embd 4096, 45 trunk layers + 1 MTP, vocabulary 154880");
    check(g.ssm_state_size == 128 && g.ssm_k_heads == 64 && g.ssm_v_heads == 64 && g.ssm_d_conv == 4 &&
              g.ssm_conv_separate && g.ssm_qk_dim == 8192 && g.ssm_value_dim == 8192 && g.ssm_conv_channels == 24576 &&
              g.kda_gate_lower_bound == -5.0f,
          "KDA: 64/64 heads x 128, conv 4 over separate q/k/v 8192 each, gate lower bound -5");
    check(g.n_head == 64 && g.n_head_kv == 1 && g.head_dim == 256 && g.head_dim_v == 256 && g.q_lora == 1536 &&
              g.kv_lora == 512 && g.n_rot == 0,
          "MLA: 64 q heads, one latent of 512, key/value 256, q LoRA 1536, no RoPE");
    check(g.idx_q_heads == 32 && g.idx_key_dim == 128 && g.idx_top_k == 2048 && g.idx_kpool == 4 &&
              g.idx_kpool_select_tail,
          "indexer: 32 heads x 128, top-k 2048, k-pool 4, tail kept (llama.cpp's default)");
    check(g.hc == 4 && g.hc_lr == 0 && g.hc_dim() == 16384 && g.hc_sinkhorn_iters == 20 &&
              std::fabs(g.hc_eps - 1e-6f) < 1e-12f,
          "mHC: 4 streams (16384), no low rank, Sinkhorn 20, eps 1e-6");
    check(g.n_expert == 288 && g.n_ff == 2048 && g.n_expert_used == 8 && g.n_expert_shared == 1 &&
              g.n_dense_lead == 3 && g.n_ff_dense == 12288,
          "MoE 288 x 2048, top-8, 1 shared; 3 dense layers of 12288");
    check(g.gating == strata::core::ExpertGating::Sigmoid && g.expert_weights_scale == 2.5f && g.expert_weights_norm &&
              g.swiglu_clamp_exp == 10.0f && g.swiglu_clamp_shexp == 10.0f,
          "sigmoid gating, scale 2.5, normalized, SwiGLU clamp 10 / 10");
    check(std::fabs(g.norm_rms_eps - 1e-5f) < 1e-11f && std::fabs(g.norm_eps - 1e-6f) < 1e-12f,
          "RMS eps 1e-5, LayerNorm eps 1e-6");
    int n_dense = 0, n_kda = 0, n_dsa = 0;
    bool kinds = true;
    for (int64_t l = 0; l < 45; ++l) {
        const LayerKind k = strata::core::layer_kind(g, l);
        const LayerKind w = l < 3 ? LayerKind::KdaDense : (l % 4 == 3 ? LayerKind::DsaMoe : LayerKind::KdaMoe);
        kinds = kinds && k == w && strata::core::is_qsa_layer(g, l) == (k == LayerKind::DsaMoe);
        n_dense += k == LayerKind::KdaDense;
        n_kda += k == LayerKind::KdaMoe;
        n_dsa += k == LayerKind::DsaMoe;
    }
    check(kinds && n_dense == 3 && n_kda == 31 && n_dsa == 11, "layers: 3 dense KDA, 31 KDA+MoE, 11 DSA+MoE at 3, 7, ..., 43");
    check(strata::core::layer_kind(g, 45) == LayerKind::DsaMoe && g.n_qsa_layers() == 11 && g.n_gdn_layers() == 34,
          "the MTP block 45 is DSA+MoE; 11 attention + 34 recurrent trunk layers");
    int64_t checked = 0;
    const bool ok = tensors_ok(dir, g, err, &checked);
    check(ok, "check_all_gguf" + (err.empty() ? std::string() : ": " + err));
    check(checked == 1412 && (size_t) checked == file_tensors(dir),
          "it compared every tensor of the file: " + std::to_string(checked) + " of 1412");
    const std::string text = strata::core::describe_geometry(g, "x: ");
    check(has(text, "x: glm5-next geometry: n_embd 4096, 45 trunk layers + 1 NextN (MTP), vocabulary 154880") &&
              has(text, "11 DSA+MoE (3 7 11 15 19 23 27 31 35 39 43)"),
          "describe_geometry prints the startup lines");
    std::fputs(text.c_str(), stdout);
}

void refusals(const fs::path& root) {
    std::printf("refusals\n");
    ModelGeometry g;
    std::string err;
    // each result is taken before the label is built: the label shows the refusal it got
    bool ok = !geometry(root / "glm-kv2", g, err) && has(err, "attention.head_count_kv[5] is 2");
    check(ok, "a layer with 2 MLA latents: " + err);
    ok = !geometry(root / "glm-embd", g, err) && has(err, "glm5-next.embedding_length = 4097, expected 4096");
    check(ok, "embedding_length 4097: " + err);
    ok = !geometry(root / "glm-clamp", g, err) && has(err, "swiglu_clamp_exp differs between layers");
    check(ok, "a per-layer clamp that differs: " + err);
    ok = geometry(root / "glm-shape", g, err) && !tensors_ok(root / "glm-shape", g, err) &&
         has(err, "layer 4: blk.4.attn_q.weight ne1 is 8191, the kernels require 8192");
    check(ok, "one tensor of the wrong shape: " + err);
    ok = geometry(root / "glm-swap", g, err) && !tensors_ok(root / "glm-swap", g, err) &&
         has(err, "layer 4 is DSA+MoE by its geometry, but blk.4.ssm_a is present");
    check(ok, "layer 4 declared DSA, its tensors KDA's: " + err);
    try {
        const strata::GgufModel m(shards_of(root / "llama"));
        strata::ModelArch a = strata::ModelArch::Qwen4Exp;
        err = strata::check_model_architecture(m.meta(), &a);
        const bool refused = has(err, "architecture is 'llama', this engine requires 'qwen4exp' or 'glm5-next'") &&
                             a == strata::ModelArch::Unknown;
        check(refused, "an unknown architecture: " + err);
        // geometry_from_gguf leaves a non-GLM file to the loaders' guard, exactly as generate.cpp did
        check(strata::core::geometry_from_gguf(m.meta(), g, err) && !g.glm() && g.n_embd == 2560,
              "geometry_from_gguf: a non-GLM file gets the Qwen defaults (its guard runs in the loaders, as before)");
    } catch (const std::exception& e) {
        check(false, std::string("llama fixture: ") + e.what());
    }
}

void loaders(const fs::path& root) {
    std::printf("native loaders (headers only: every check before the payload)\n");
    for (const char* name : {"glm", "qwen"}) {
        ModelGeometry g;
        std::string err;
        geometry(root / name, g, err);
        const std::vector<std::string> shards = shards_of(root / name);
        {
            strata::core::NativeHead head;
            const bool loaded = head.load(shards, g.n_embd, g.n_vocab, err);
            const bool payload = err == "native head: truncated output.weight payload";
            check(!loaded && payload,
                  std::string(name) + ": NativeHead takes output.weight [" + std::to_string(g.n_embd) + ", " +
                      std::to_string(g.n_vocab) + "] Q6_K up to the payload: " + err);
        }
        {
            strata::core::NativeEmbed embed;
            const bool loaded = embed.load(shards, g.n_embd, g.n_vocab, err);
            check(!loaded && has(err, "native embedding: token_embd.weight's payload is truncated"),
                  std::string(name) + ": NativeEmbed takes token_embd.weight Q8_0 up to the payload");
        }
        std::set<std::string> served;
        check(strata::core::NativeDense::served_names(shards, false, served, err), std::string(name) + ": served_names");
        int dense = 0;
        for (const auto& n : served)
            dense += n.ends_with(".ffn_gate.weight") || n.ends_with(".ffn_up.weight") || n.ends_with(".ffn_down.weight");
        if (std::string(name) == "glm")
            check(served.size() == 298 && dense == 9 && served.count("blk.0.ffn_gate.weight") &&
                      served.count("blk.2.ffn_down.weight"),
                  "glm: NativeDense serves 298 matrices, the dense FFN of layers 0-2 (9) among them");
        else
            check(served.size() == 300 && dense == 0, "qwen: NativeDense serves the same 300 matrices as before");
    }
    {
        strata::core::NativeHead head;
        std::string err;
        head.load(shards_of(root / "llama"), 2560, 248320, err);
        const bool refused = has(err, "architecture is 'llama'");
        check(refused, "NativeHead refuses an unknown architecture: " + err);
    }
}

void native_experts(const fs::path& tmp) {
#if defined(STRATA_NATIVE_EXPERTS)
    std::printf("native expert formats at the geometry's widths\n");
    namespace cpu = strata::kernels::cpu;
    constexpr int IQ3_S = 21, IQ4_XS = 23, Q6_K = 14, IQ4_NL = 20;
    struct Pair {
        int gu, d;
        size_t bytes;
        const char* what;
    };
    const Pair pairs[] = {
        {IQ3_S, IQ4_XS, 11665408, "GLM IQ3_S/IQ3_S/IQ4_XS (39 layers): 3,604,480 x 2 + 4,456,448"},
        {IQ3_S, Q6_K, 14090240, "GLM IQ3_S/IQ3_S/Q6_K (layers 12, 44)"},
        {IQ4_XS, Q6_K, 15794176, "GLM IQ4_XS/IQ4_XS/Q6_K (layer 11)"},
    };
    for (const Pair& p : pairs) {
        cpu::NativeFmt f;
        std::string err;
        const bool ok = cpu::native_fmt(p.gu, p.d, 4096, 2048, f, err);
        check(ok && f.bytes == p.bytes && f.act_bytes <= cpu::kNativeActBytes && f.h_bytes <= cpu::kNativeHBytes,
              std::string(p.what) + ": " + std::to_string(f.bytes) + " B" + (err.empty() ? "" : " - " + err));
    }
    {
        cpu::NativeFmt f;
        std::string err;
        check(cpu::native_fmt(IQ3_S, IQ4_NL, 2560, 640, f, err) && f.bytes == 2329600 && f.act_bytes == 2920 &&
                  f.h_bytes == 680,
              "Qwen IQ3_S/IQ3_S/IQ4_NL at 2560/640: 2,329,600 B, activations 2,920 / 680 B");
    }
    // a GLM pack's native_experts.txt: layers 0-2 dense (no line), 3-44 IQ3_S/IQ4_XS, layer 11 IQ4_XS/Q6_K
    fs::create_directories(tmp);
    {
        std::ofstream o(tmp / "native_experts.txt", std::ios::binary);
        o << "# strata native experts v3: layer gate_up_type down_type offset blob_bytes (n_expert 288)\n";
        uint64_t off = 0;
        for (int l = 3; l < 45; ++l) {
            const int gu = l == 11 ? IQ4_XS : IQ3_S, d = (l == 11 || l == 12 || l == 44) ? Q6_K : IQ4_XS;
            const uint64_t blob = l == 11 ? 15794176ull : d == Q6_K ? 14090240ull : 11665408ull;
            o << l << " " << gu << " " << d << " " << off << " " << blob << "\n";
            off += blob * 288;
        }
    }
    std::string err;
    const bool ok = cpu::expert_layout_load(tmp.string(), 45, 288, err, 4096, 2048, 3);
    const auto& lay = cpu::expert_layout();
    // 39 x 11,665,408 + 2 x 14,090,240 + 15,794,176 = 498,925,568 B per expert, all layers; x 288 experts
    check(ok && lay.native && lay.bytes[0] == 0 && lay.bytes[2] == 0 && lay.layer_offset(3) == 0 &&
              lay.max_blob == 15794176,
          "a 45-layer GLM layout with 3 dense layers" + (err.empty() ? std::string() : " - " + err));
    check(lay.total == 143690563584ull,
          "its experts total " + std::to_string(lay.total) + " B = the plan's 143,690,563,584 (1.1, from the GGUF)");
    const bool wide = cpu::expert_layout_load(tmp.string(), 45, 288, err, 2560, 640, 3);
    const std::string why = err;
    check(!wide, "the same file at Qwen's widths is refused: " + why);
    // restore the canonical layout for anything after this
    cpu::expert_layout_load((tmp / "none").string(), 48, 512, err);
#else
    (void) tmp;
    std::printf("native expert formats: skipped (built without STRATA_NATIVE_EXPERTS)\n");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: model_geometry_test FIXTURE_DIR (written by tests/core/geometry_fixtures.py)\n");
        return 2;
    }
    const fs::path root = argv[1];
    std::printf("model_geometry_test %s\n", root.string().c_str());
    try {
        qwen(root);
        glm(root);
        refusals(root);
        loaders(root);
        native_experts(fs::temp_directory_path() /
                       ("strata-model-geometry-test-" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())));
    } catch (const std::exception& e) {
        check(false, std::string("unexpected exception: ") + e.what());
    }
    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
