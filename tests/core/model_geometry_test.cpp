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
//   6. GLM-5.3-Flash UD-IQ3_XXS (glm-iq3, read on 2026-10-07): 4 shards (split keys, which shard holds which tensor,
//      a shard with a wrong split.no refused), the same geometry as UD-IQ4_XS, every tensor checked, the per-layer
//      expert formats (IQ2_S / IQ3_S, IQ2_S / IQ4_XS at 12 and 44, IQ3_S / IQ4_XS at 11) with GPU kernels and the
//      prompt path's dequantizer, 109,867,696,128 B of experts; the loaders as in 4 with a Q6_K token_embd.
#include "gguf_fixture.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <array>
#include <cstring>
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

/// GLM-5.3-Flash UD-IQ3_XXS: the same model as UD-IQ4_XS in a smaller quant, split in 4 (shard 1 metadata only)
void glm_iq3(const fs::path& root) {
    std::printf("glm5-next UD-IQ3_XXS (4 shards)\n");
    const fs::path dir = root / "glm-iq3";
    const std::vector<std::string> shards = shards_of(dir);
    check(shards.size() == 4 && shards[0].ends_with("GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf") &&
              shards[3].ends_with("-00004-of-00004.gguf"),
          "the split names resolve to 4 shards, the metadata shard first");
    const strata::GgufModel m(shards);
    bool keys = true;
    std::array<size_t, 4> per{};
    for (size_t i = 0; i < m.size() && i < 4; ++i) {
        const strata::MetaValue* c = m.shard(i).get("split.count");
        const strata::MetaValue* n = m.shard(i).get("split.no");
        const strata::MetaValue* t = m.shard(i).get("split.tensors.count");
        keys = keys && c && n && t && c->u == 4 && n->u == i && t->u == 1412;
        per[i] = m.shard(i).tensors().size();
    }
    check(keys && per == std::array<size_t, 4>{0, 614, 558, 240} && m.meta().get("general.architecture") &&
              m.shard(1).get("general.architecture") == nullptr,
          "split.count 4, split.no 0-3, split.tensors.count 1412 in every shard; tensors 0 / 614 / 558 / 240; "
          "the metadata in shard 1 only");
    // where the tensors are: output and token_embd in shard 2; layer l's three expert tensors in one shard, 2 for
    // layers 3-19, 3 for 20-37, 4 for 38-45 (blocks 20 and 38 straddle a boundary: their other tensors start earlier)
    auto shard_of = [&](const std::string& name) {
        size_t at = 99;
        return m.find(name, &at) ? (int) at : -1;
    };
    bool where = shard_of("output.weight") == 1 && shard_of("token_embd.weight") == 1 &&
                 shard_of("blk.20.attn_norm.weight") == 1 && shard_of("blk.38.attn_norm.weight") == 2;
    const char* roles[3] = {"gate", "up", "down"};
    for (int l = 3; l <= 45; ++l)
        for (const char* r : roles)
            where = where && shard_of("blk." + std::to_string(l) + ".ffn_" + r + "_exps.weight") ==
                                 (l < 20 ? 1 : l < 38 ? 2 : 3);
    check(where, "output / token_embd in shard 2; experts of layers 3-19 in shard 2, 20-37 in 3, 38-45 in 4");
    ModelGeometry g, g4;
    std::string err, err4;
    const bool read = geometry(dir, g, err) && geometry(root / "glm", g4, err4);
    check(read && strata::core::describe_geometry(g, "x: ") == strata::core::describe_geometry(g4, "x: ") &&
              g.n_layers == 45 && g.n_expert == 288 && g.n_ff == 2048 && g.n_dense_lead == 3,
          "geometry_from_gguf: the UD-IQ4_XS geometry, line for line" + (err.empty() ? std::string() : ": " + err));
    int64_t checked = 0;
    const bool ok = tensors_ok(dir, g, err, &checked);
    check(ok && checked == 1412, "check_all_gguf: " + std::to_string(checked) + " of 1412 tensors" +
                                     (err.empty() ? std::string() : ": " + err));
    // the expert formats per layer, and the trunk's expert bytes (blk.45, the MTP block, is left out by the pack)
    constexpr uint32_t IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, Q2_K = 10, Q3_K = 11;
    bool fmts = true, kernels = true;
    uint64_t total = 0;
    for (int l = 3; l <= 45; ++l) {
        const auto* gt = m.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
        const auto* ut = m.find("blk." + std::to_string(l) + ".ffn_up_exps.weight");
        const auto* dt = m.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
        if (!gt || !ut || !dt) { fmts = false; continue; }
        const uint32_t wg = l == 45 ? Q2_K : l == 11 ? IQ3_S : IQ2_S;
        const uint32_t wd = l == 45 ? Q3_K : (l == 11 || l == 12 || l == 44) ? IQ4_XS : IQ3_S;
        fmts = fmts && (uint32_t) gt->type == wg && (uint32_t) ut->type == wg && (uint32_t) dt->type == wd;
        if (l == 45) continue;
        total += strata::tensor_payload_bytes(*gt) + strata::tensor_payload_bytes(*ut) + strata::tensor_payload_bytes(*dt);
#if defined(STRATA_NATIVE_EXPERTS)
        kernels = kernels && strata::kernels::native_expert_supported((int) gt->type, (int) dt->type, g.n_embd, g.n_ff);
#endif
    }
    check(fmts, "experts: IQ2_S/IQ2_S/IQ3_S in 39 layers, IQ2_S/IQ2_S/IQ4_XS at 12 and 44, IQ3_S/IQ3_S/IQ4_XS at 11 "
                "(MTP 45: Q2_K/Q2_K/Q3_K)");
#if defined(STRATA_NATIVE_EXPERTS)
    check(kernels, "native_expert_supported (GPU grouped kernels + the prompt path's FP16 dequantizer) for every layer");
#endif
    check(total == 109867696128ull, "the trunk's experts (layers 3-44): " + std::to_string(total) +
                                        " B = 109,867,696,128 (102.32 GiB)");
    // a shard that declares another place in the split
    const bool refused = !geometry(root / "glm-iq3-split", g, err) &&
                         has(err, "does not declare itself shard 3 of 4");
    check(refused, "a shard with split.no 3 in third place is refused: " + err);
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
    ok = !geometry(root / "glm-blocks", g, err) &&
         has(err, "glm5-next.block_count = 47, expected 46 (a file with a NextN block is the whole model)");
    check(ok, "the whole model (it has a NextN block) with 47 blocks: " + err);
    ok = !geometry(root / "glm-slice8-kv", g, err) &&
         has(err, "attention.head_count_kv must be a per-layer array of 8 entries");
    check(ok, "a trunk slice whose head_count_kv has 9 entries: " + err);
    // a trunk slice (validate's glm-mini: blocks 0-7, no NextN block): accepted, its geometry and tensors checked
    ok = geometry(root / "glm-slice8", g, err);
    check(ok && g.glm() && g.n_layers == 8 && g.n_layers_mtp == 0 && g.n_dense_lead == 3 && g.n_qsa_layers() == 2 &&
              strata::core::layer_kind(g, 3) == LayerKind::DsaMoe && strata::core::layer_kind(g, 7) == LayerKind::DsaMoe &&
              strata::core::layer_kind(g, 2) == LayerKind::KdaDense && strata::core::layer_kind(g, 4) == LayerKind::KdaMoe,
          "a trunk slice of 8 blocks: 3 dense KDA, DSA at 3 and 7" + (err.empty() ? std::string() : ": " + err));
    int64_t checked = 0;
    ok = tensors_ok(root / "glm-slice8", g, err, &checked);
    check(ok && (size_t) checked == file_tensors(root / "glm-slice8"),
          "check_all_gguf on the slice: " + std::to_string(checked) + " tensors" + (ok ? "" : ": " + err));
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
    std::set<std::string> served_glm;
    for (const char* name : {"glm", "qwen", "glm-iq3"}) {
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
            const bool q6 = std::string(name) == "glm-iq3";   // UD-IQ3_XXS: Q6_K (glm_embed_type_supported)
            check(!loaded && has(err, "native embedding: token_embd.weight's payload is truncated"),
                  std::string(name) + ": NativeEmbed takes token_embd.weight " + (q6 ? "Q6_K" : "Q8_0") +
                      " up to the payload" + (loaded ? std::string() : ": " + err));
        }
        std::set<std::string> served;
        check(strata::core::NativeDense::served_names(shards, false, served, err), std::string(name) + ": served_names");
        int dense = 0;
        for (const auto& n : served)
            dense += n.ends_with(".ffn_gate.weight") || n.ends_with(".ffn_up.weight") || n.ends_with(".ffn_down.weight");
        if (std::string(name) == "glm") served_glm = served;
        if (std::string(name) == "glm-iq3")
            check(served == served_glm, "glm-iq3: NativeDense serves the same 643 matrices as UD-IQ4_XS (Q8_0 or Q6_K)");
        else if (std::string(name) == "glm")
            check(served.size() == 643 && dense == 9 && served.count("blk.0.ffn_gate.weight") &&
                      served.count("blk.2.ffn_down.weight") && served.count("blk.3.attn_k_b.weight") &&
                      served.count("blk.0.hc_attn_fn.weight") && !served.count("blk.3.ffn_gate_inp.weight") &&
                      !served.count("blk.3.ffn_gate_exps.weight"),
                  "glm: NativeDense serves every Q8_0 non-expert matrix (643: 630 trunk + 13 MTP; 3-D k_b/v_b "
                  "flattened; the dense FFN's 9 among them), no F32 row, no expert");
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

#if defined(STRATA_NATIVE_EXPERTS)
/// A layer without experts (GLM's dense 0-2) through the whole file tier, on a 3-layer synthetic model whose layer 0
/// is dense: the layout, check_experts_gguf, load_experts_gguf and FileExpertSource reading the GGUF in place.
void dense_layers_in_place(const fs::path& dir) {
    namespace cpu = strata::kernels::cpu;
    using fixture::Tensor;
    constexpr int64_t E = 256, FF = 256, NE = 2;
    constexpr uint32_t Q8_0 = 8;
    fs::create_directories(dir / "pack");
    const fs::path gguf = dir / "Mini.gguf";
    const std::vector<Tensor> ts = {{"blk.0.ffn_gate.weight", {E, 512}, Q8_0, 1},
                                    {"blk.1.ffn_gate_exps.weight", {E, FF, NE}, Q8_0, 11},
                                    {"blk.1.ffn_up_exps.weight", {E, FF, NE}, Q8_0, 12},
                                    {"blk.1.ffn_down_exps.weight", {FF, E, NE}, Q8_0, 13},
                                    {"blk.2.ffn_gate_exps.weight", {E, FF, NE}, Q8_0, 21},
                                    {"blk.2.ffn_up_exps.weight", {E, FF, NE}, Q8_0, 22},
                                    {"blk.2.ffn_down_exps.weight", {FF, E, NE}, Q8_0, 23}};
    const fixture::Written w = fixture::write(gguf, {fixture::str("general.architecture", "mini")}, ts);
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) Q8_0, (int) Q8_0, E, FF, f, err)) { check(false, "Q8_0 at 256: " + err); return; }
    auto at = [&](size_t i) { return std::to_string(w.data_start + w.offsets[i]); };
    {
        std::ofstream o(dir / "pack" / "native_experts.txt", std::ios::binary);
        o << "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off (n_expert 2)\n";
        o << "1 8 8 0 " << f.bytes << " " << at(1) << " " << at(2) << " " << at(3) << "\n";
        o << "2 8 8 " << f.bytes * NE << " " << f.bytes << " " << at(4) << " " << at(5) << " " << at(6) << "\n";
    }
    const std::string pack = (dir / "pack").string();
    bool ok = !cpu::expert_layout_load(pack, 3, NE, err, E, FF, 0);
    check(ok && has(err, "layer 0 is missing"), "without a dense lead, the missing layer 0 is refused: " + err);
    err.clear();
    ok = cpu::expert_layout_load(pack, 3, NE, err, E, FF, 1);
    const cpu::ExpertLayout& L = cpu::expert_layout();
    check(ok && !L.has_experts(0) && L.has_experts(1) && L.has_experts(2) && L.blob_bytes(0) == 0 &&
              L.layer_offset(1) == 0 && L.total == 2 * NE * f.bytes,
          "dense lead 1: layer 0 has no experts, layers 1-2 follow from offset 0" + (err.empty() ? "" : ": " + err));
    if (!ok) return;
    ok = strata::core::check_experts_gguf(gguf.string(), L, err);
    check(ok, "check_experts_gguf skips the dense layer" + (ok ? std::string() : ": " + err));
    std::vector<uint8_t> arena((size_t) L.total, 0xEE);
    const strata::core::LoadStats st = strata::core::load_experts_gguf(gguf.string(), arena.data(), L, 2);
    bool exact = st.ok && st.bytes == L.total;
    for (int64_t l = 1; l < 3 && exact; ++l)
        for (int64_t e = 0; e < NE && exact; ++e) {
            const uint64_t per[3] = {f.up_off, f.up_off, f.bytes - f.down_off}, off[3] = {0, f.up_off, f.down_off};
            for (int r = 0; r < 3 && exact; ++r)
                for (uint64_t j = 0; j < per[r]; ++j)
                    if (arena[(size_t) (L.blob_offset(l, e) + off[r] + j)] !=
                        fixture::pattern((uint8_t) (10 * l + 1 + r), (uint64_t) e * per[r] + j)) { exact = false; break; }
        }
    check(exact, "load_experts_gguf: every blob of layers 1-2 is its tensors' bytes");
    strata::core::FileExpertSource src;
    src.set_gguf(gguf.string());
    ok = src.open(pack, 3, NE, err);
    bool same = ok && src.gguf_mode();
    for (int64_t l = 1; l < 3 && same; ++l)
        for (int64_t e = 0; e < NE && same; ++e) {
            src.begin_layer(l, nullptr, 0);
            const uint8_t* b = src.blob(l, e);
            same = b != nullptr && std::memcmp(b, arena.data() + L.blob_offset(l, e), (size_t) f.bytes) == 0;
        }
    check(same, "FileExpertSource opens a layout with a dense layer and reads layers 1-2 in place" +
                    (ok ? std::string() : ": " + err));
}
#endif

void native_experts(const fs::path& tmp) {
#if defined(STRATA_NATIVE_EXPERTS)
    std::printf("native expert formats at the geometry's widths\n");
    namespace cpu = strata::kernels::cpu;
    constexpr int IQ3_S = 21, IQ4_XS = 23, Q6_K = 14, IQ4_NL = 20, IQ2_S = 22;
    struct Pair {
        int gu, d;
        size_t bytes;
        const char* what;
    };
    const Pair pairs[] = {
        {IQ3_S, IQ4_XS, 11665408, "GLM IQ3_S/IQ3_S/IQ4_XS (39 layers): 3,604,480 x 2 + 4,456,448"},
        {IQ3_S, Q6_K, 14090240, "GLM IQ3_S/IQ3_S/Q6_K (layers 12, 44)"},
        {IQ4_XS, Q6_K, 15794176, "GLM IQ4_XS/IQ4_XS/Q6_K (layer 11)"},
        {IQ2_S, IQ3_S, 8978432, "GLM UD-IQ3_XXS IQ2_S/IQ2_S/IQ3_S (39 layers): 2,686,976 x 2 + 3,604,480"},
        {IQ2_S, IQ4_XS, 9830400, "GLM UD-IQ3_XXS IQ2_S/IQ2_S/IQ4_XS (layers 12, 44)"},
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
    // UD-IQ3_XXS's layout (tools/iq_pack.py writes these types and blob sizes from the GGUF)
    {
        std::ofstream o(tmp / "native_experts.txt", std::ios::binary);
        o << "# strata native experts v3: layer gate_up_type down_type offset blob_bytes (n_expert 288)\n";
        uint64_t off = 0;
        for (int l = 3; l < 45; ++l) {
            const int gu = l == 11 ? IQ3_S : IQ2_S, d = (l == 11 || l == 12 || l == 44) ? IQ4_XS : IQ3_S;
            const uint64_t blob = l == 11 ? 11665408ull : d == IQ4_XS ? 9830400ull : 8978432ull;
            o << l << " " << gu << " " << d << " " << off << " " << blob << "\n";
            off += blob * 288;
        }
    }
    err.clear();
    const bool ok3 = cpu::expert_layout_load(tmp.string(), 45, 288, err, 4096, 2048, 3);
    check(ok3 && lay.native && lay.bytes[2] == 0 && lay.total == 109867696128ull && lay.max_blob == 11665408 &&
              lay.fmt[3].gu_type == IQ2_S && lay.fmt[3].d_type == IQ3_S && lay.fmt[12].d_type == IQ4_XS,
          "UD-IQ3_XXS's 45-layer layout: " + std::to_string(lay.total) + " B = 109,867,696,128" +
              (err.empty() ? std::string() : " - " + err));
    dense_layers_in_place(tmp / "mini");
    // restore the canonical layout for anything after this
    cpu::expert_layout_load((tmp / "none").string(), 48, 512, err);
#else
    (void) tmp;
    std::printf("native expert formats: skipped (built without STRATA_NATIVE_EXPERTS)\n");
#endif
}

/// `model_geometry_test --pack DIR --native SHARD`: a real pack (tools/iq_pack.py) against its model file - headers,
/// native_experts.txt and index.txt only.  Not a ctest: it needs the pack and the shards (sparse copies do).
int real_pack(const std::string& pack, const std::string& shard) {
    namespace cpu = strata::kernels::cpu;
    std::string err;
    const std::vector<std::string> shards = strata::gguf_split_paths(shard);
    const strata::GgufModel m(shards);
    ModelGeometry g;
    bool ok = strata::core::geometry_from_gguf(m.meta(), g, err);
    check(ok, "geometry_from_gguf" + (ok ? std::string() : ": " + err));
    int64_t checked = 0;
    ok = strata::core::check_all_gguf(m, g, err, &checked);
    check(ok, "check_all_gguf: " + std::to_string(checked) + " tensors" + (ok ? std::string() : ": " + err));
    err.clear();
    ok = cpu::expert_layout_load(pack, g.n_layers, g.n_expert, err, g.n_embd, g.n_ff, g.n_dense_lead);
    const cpu::ExpertLayout& L = cpu::expert_layout();
    check(ok && L.native, "native_experts.txt at the geometry's widths: " + std::to_string(L.total) + " B, largest blob " +
                              std::to_string(L.max_blob) + (ok ? "" : ": " + err));
    if (ok) {
        ok = strata::core::check_experts_gguf(shards.front(), L, err);
        check(ok, "check_experts_gguf: every layer's expert spans are its tensors" + (ok ? std::string() : ": " + err));
        strata::core::FileExpertSource src;
        src.set_gguf(shards.front());
        ok = src.open(pack, g.n_layers, L.n_expert, err);
        check(ok && src.gguf_mode(), "FileExpertSource opens the shards in place" + (ok ? std::string() : ": " + err));
    }
    // index.txt rows (name, file, kind, ...) against what NativeDense serves from the GGUF
    std::set<std::string> served, rows, quantized_rows;
    strata::core::NativeDense::served_names(shards, false, served, err);
    std::ifstream in(pack + "/index.txt");
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        char name[256];
        int file = 0, kind = -1;
        if (std::sscanf(line.c_str(), "%255s %d %d", name, &file, &kind) != 3) continue;
        rows.insert(name);
        if (kind == 0 && std::string(name).rfind("blk.", 0) == 0) quantized_rows.insert(name);
    }
    int64_t missing = 0, extra = 0;
    for (const auto& n : quantized_rows) missing += served.count(n) == 0;
    for (const auto& n : served) extra += rows.count(n) == 0;
    // glm5-next's pack leaves every quantized non-expert tensor in GGUF form (no bytes in dense.bin): all must be
    // served.  A Qwen pack keeps some quantized rows in the arena (the PLE key unless --native-ple-key)
    const std::string what = std::to_string(quantized_rows.size()) + " quantized blk rows of index.txt, " +
                             std::to_string(quantized_rows.size() - (size_t) missing) + " served natively";
    if (g.glm()) check(missing == 0, what);
    else std::printf("  %s\n", what.c_str());
    std::printf("  served names absent from index.txt (left out by the pack, e.g. the MTP block): %lld\n",
                (long long) extra);
    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "--pack" && std::string(argv[3]) == "--native") {
        try {
            return real_pack(argv[2], argv[4]);
        } catch (const std::exception& e) {
            std::printf("refused: %s\n", e.what());
            return 1;
        }
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: model_geometry_test FIXTURE_DIR (written by tests/core/geometry_fixtures.py)\n");
        return 2;
    }
    const fs::path root = argv[1];
    std::printf("model_geometry_test %s\n", root.string().c_str());
    try {
        qwen(root);
        glm(root);
        glm_iq3(root);
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
