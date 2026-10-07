# GLM-5.3-Flash (`glm5-next`)

Strata's engine and server also run **GLM-5.3-Flash**, llama.cpp's architecture `glm5-next` (45 trunk layers: 34 KDA
linear-attention layers and 11 DSA/MLA attention layers, mHC hyper-connections, 288 routed experts with sigmoid
gating, top-8, plus one shared expert; 3 dense leading layers; a NextN/MTP layer that is not loaded). The reference
for every operation is llama.cpp **f1cee99** (b11342), `src/models/glm5-next.cpp`; the parity tests compare against
ggml-cpu built from that commit.

The Qwen path is not changed: a `qwen4exp` file takes exactly the code it took before, and a `glm5-next` file goes to
a separate path (`src/program/glm_generate.cpp`, `src/core/glm_*.cpp`, `src/prefill/glm_*`, `src/kernels/cuda/` new
files). Shared files gain only the geometry read from the model file, shape limits wide enough for both models, and
options that are off by default.

setup.py does not install GLM; the run config is written by hand (`serve/examples/strata-glm-ud-iq3_xxs.json`).

## What works

| Part | State |
|---|---|
| Loading | The architecture guard, `ModelGeometry` from the GGUF keys, every tensor of the file checked against the geometry before anything is allocated, the pack (`tools/iq_pack.py`) and the tokenizer (`glm4` pre-tokenizer) |
| Decode | Verify windows of 1-8 tokens (`--spec T`, default 1) with `commit(n)`; no drafter, so decode is one token per window |
| Prompts | The prompt path in chunks (`--prefill N\|auto`): KDA over the chunk, the DSA prompt attention, routed experts as FP16 GEMMs per expert |
| Expert tiers | A VRAM cache (`--expert-cache N\|auto`), a page-locked RAM tier (`--resident-budget-gib`), the rest read from the GGUF by a reader pool with a one-layer lookahead (O_DIRECT by default) |
| RAM-tier experts | Staged over PCIe and computed on the GPU, or computed by the CPU expert pool (`--glm-pool on\|off\|auto`) |
| Serving | `--serve` with the Qwen serve loop's protocol: sampling, stop ids, streaming, conversation checkpoints (the KDA state and the DSA cache), parking in RAM, prefix reuse, `--kv-resident` streaming of the DSA cache |
| Server | A per-model profile chosen from the pack's tokenizer: GLM's chat template, end tokens, thinking on/off and effort levels, GLM's tool-call grammar parsed into OpenAI / Anthropic / Responses tool calls, streamed |

## Measured

On one **RTX 3090** (CUDA 13.4, sm_86) in a server with an AMD EPYC 9654P (24 logical CPUs given to the engine), 94 GiB
of RAM on one DIMM and an NVMe RAID that reads cold 11.7 MB expert blobs at 3.68 GB/s with 8 threads; Linux. Model:
Unsloth's **GLM-5.3-Flash UD-IQ3_XXS** (`unsloth/GLM-5.3-Flash-GGUF`, revision `a38483c`, 4 shards, 112 GiB; 102.3 GiB
of routed experts, 3.02 GB of routed bytes per decode token). llama.cpp is f1cee99 on the same file.

### Agreement with llama.cpp

Positions of greedy answers, teacher-forced through both engines; same top token / top-20 KL. The floor is
llama.cpp's own CUDA build against its CPU build on the same positions.

| Set | Strata vs llama.cpp CUDA | Strata vs llama.cpp CPU | Floor: llama.cpp CUDA vs CPU |
|---|---|---|---|
| code answer, 460 positions | 97.61% / 0.00897 | | 96.96% / 0.01034 |
| thinking answer, 204 | 94.61% / 0.03068 | | 92.65% / 0.04396 |
| after a 2,137-token prompt, 191 | 96.3% / 0.015 | 94.8% / 0.028 | 94.2% / 0.032 |
| 4K prompt, 63 rows after it | 87.3% / 0.147 | 82.5% / 0.240 | 82.5% / 0.229 |
| 8K prompt, 63 rows | 85.7% / 0.095 | 85.7% / 0.141 | 85.7% / 0.131 |
| 16K needle prompt, 63 rows of its tail | 93.7% / 0.064 | 88.9% / 0.067 | 92.1% / 0.071 |
| 16K needle prompt, the 12 answer rows | 100% / 3.3e-4 | 100% / 5.7e-5 | 100% / 3.9e-4 |

Strata's rows are as close to llama.cpp CPU as llama.cpp CUDA is, at every length; both found the needle at 16K. The
4K and 8K rows are high-perplexity document text, where llama.cpp's two builds disagree as much. The DSA selection
(2,048 of the pooled cells per query past 2,051 tokens) overlaps llama.cpp CPU's by 99.2% (layer 3) to ~98% (layer
15); against llama.cpp CUDA it is 91-95% in the deep layers, because llama.cpp's CUDA indexer rounds queries and
keys to f16 and Strata's follows ggml-cpu's f32.

### Speed

| | Strata | llama.cpp, same file, same machine |
|---|---|---|
| Decode, varied text (code, prose, Japanese, an agent-session excerpt) | **7.05 tok/s** and 7.78 tok/s (two runs) | 3.01 (a code answer) to 4.62 tok/s (a thinking answer), 2.61 after a 16K prompt (cold page cache) |
| Decode, the same code answer read a second time | 6.28 tok/s | |
| Prompt, 16,313 tokens | **274.8 tok/s** (`--prefill auto`: 8,192-token chunks) | 256.7 tok/s |
| Prompt, 8,192 / 4,096-token chunks on the same prompt | 236.0 / 167.2 tok/s | |
| Engine load (RAM tier filled, 32K context) | 55-65 s | 73 s |

Decode setup: `--expert-cache auto` (1,606-1,692 experts in VRAM at 32K context), `--resident-budget-gib 67` (7,912
experts page-locked in RAM), the rest from the file tier, `--glm-pool off`. On the varied set 27% of expert lookups
came from VRAM, 60% from RAM, 12% from the file tier (283 MB per token at 3.76 GB/s through the reader pool).

Through the HTTP server at 262,144 context (`serve/examples/strata-glm-ud-iq3_xxs.json`; 1,344 experts in VRAM beside
the 706 MiB session and 2,862 MiB of prompt buffers): decode 5.2-6.6 tok/s, a 16,019-token prompt in 101 s
including its system and turn parts, healthy 61 s after start; a tool-call round trip, three-turn prefix reuse
and thinking off all worked (`tests/glm/serve_e2e.py`: 32 of 32 checks on the real model; that run also had a
conversation-cache disk tier, a separate change, whose checks this tree's version of the script leaves out).

### Qwen

Qwen3.8-Flash-Next UD-IQ4_XS on the same machine: `STRATA_LOGPOS` top-20 rows of two sets (865 rows) are
byte-identical to the unchanged engine's when the expert cache is pinned to the same slot count (`--expert-cache
4850`). With `--expert-cache auto` the larger binary (the GLM kernels are loaded eagerly before the cache is sized
from free VRAM) leaves 4,849 slots instead of 4,850 on the 3090: one expert moves to the RAM tier, the CPU rounds it
differently, and the rows drift at rounding level (same top token 100% / 99.0%, mean top-20 KL 2.3e-4 / 1.4e-3).

### Development GPU

The kernels were written and parity-tested on an **RX 9070 XT** (gfx1201, ROCm 7.2.4, HIP build) with a Ryzen 9
7950X, then confirmed on the 3090. "glm-mini" in comments is a real-weight slice of the GGUF used there: blocks 0-7
(3 dense KDA layers, 2 DSA+MoE, 3 KDA+MoE), the first 32 experts of each MoE layer, the vocabulary tensors whole
(4.84 GB); llama.cpp loads it as an 8-block trunk. On it the whole 8-layer model through Strata's head agreed with
llama.cpp CPU at 95.3% / top-20 KL 8.7e-5 (code) where llama.cpp's own HIP build agreed at 95.4% / 7.9e-5.

## Not done

- **No MTP drafter.** The GGUF's NextN layer is not loaded (its 3.2 GiB of experts would not fit beside the cache),
  so `--mtp` and drafts do not apply and decode is one token per window.
- **One GPU.** The GLM path has no layer split and no helper cards' expert caches (and Strata's pinned RAM tier
  refuses both for Qwen too).
- **`--glm-pool auto` can pick the slower arm.** It times the two arms in isolation at start. On the EPYC above it
  picked the CPU pool and decoded at 5.90 tok/s against 7.05 with staging (`--glm-pool off`): 23 Zen 4 workers read
  RAM-tier experts at 16.6 GB/s, PCIe staging moved 26.9 GB/s. On the 7950X with glm-mini the pool was 9x faster than
  staging. Measure both on a new machine.
- **Short prompt parts are slow.** A prompt part of at most `--short-read` tokens goes through decode windows of one
  token; a longer one through the prompt path, whose first chunk streams every expert not in VRAM. A 25-189 token
  request took 5-18 s to its first token in service.
- **No adaptive expert tier.** The VRAM cache and the RAM tier are filled once, in the profile's order;
  `--expert-profile-save` saves nothing on this path (the engine says so).
- **`effort_position: "end"`** is refused for GLM until `--tail-role-token` handles GLM's role tokens.
- **Session files** (`--slot-save-path`, `POST /slots/0?action=save|restore`) are refused for GLM (`SERR invalid`,
  a 400): their format holds the Qwen layers' state. Parked conversations in RAM work as for Qwen.
- A request's `pin=N` (`"strata_prefix"`) is ignored on the GLM path (`ckpt=0`, `"strata_checkpoint": false`, is
  honoured), and the opt-in `"tool_call_recovery"` is written for the Qwen template's call form and is off for GLM.
- Built and measured on Linux only. On Windows the file tier's reader copies from the mapping (no `O_DIRECT`);
  not built or run there.

## How to run

```bash
# 1. the model (4 shards) - any download that gives the files of revision a38483c
hf download unsloth/GLM-5.3-Flash-GGUF --revision a38483c8cd5df544f53d70fb281afe97369d5ab6 \
   --include "UD-IQ3_XXS/*" --local-dir data/models/GLM-5.3-Flash-GGUF

# 2. the pack: index, F32 tensors, tokenizer and the expert table; the experts stay in the GGUF
python tools/iq_pack.py --gguf data/models/GLM-5.3-Flash-GGUF/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
   --out data/packs/glm-ud-iq3_xxs

# 3. the engine alone: teacher-forced windows over token ids (one id per line or comma-separated)
<build>/strata --pack data/packs/glm-ud-iq3_xxs \
   --native data/models/GLM-5.3-Flash-GGUF/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
   --tokens-file ids.txt --expert-cache auto --resident-budget-gib 67 --prefill auto --max-new 64 --stats

# 4. the server
python -m serve.server --engine strata --config strata-glm-ud-iq3_xxs.json --port 8080
```

Size `--resident-budget-gib` with setup's rule, RAM - 24 - ceil(KV RAM in GB); the 67 above is for 94 GiB of RAM at
262,144 context. A parked GLM conversation is 152.6 MB of KDA state plus 12,320 B per token (int8 DSA cache), so the
conversation cache's budget is smaller than Qwen's. The server adds `--eos-ids 154820,154827,154829 --turn-token
154828` for a GLM pack. `--kv` is `int8` for GLM (`f16` keeps the DSA latent in f16: no closer to llama.cpp at 16K,
185 instead of 275 tok/s prompt).

An expert profile (`--expert-profile`, the STRP format of `tools/make_profile.py`) is optional: without one the VRAM
cache and the RAM tier fill in layer-interleaved index order. `make_profile.py --gguf <shard 1>` takes GLM's
dimensions (45 x 288); its traces are routing records in `--dump-routing`'s format, which for GLM came from llama.cpp
(an eval-callback dump of `ffn_moe_topk-N`; that tool is not part of this tree). GLM's routing is close to uniform over
short windows (10,400 of the 12,096 trunk experts are touched in 256 tokens), so the profile matters less than Qwen's.

## Tests

CPU only, no model: `model_geometry_test` (the guard, the geometry and every tensor check on header-only GGUFs written
from `tests/data/gguf_headers/*.json` by `tools/gguf_header_fixture.py`: GLM UD-IQ4_XS, UD-IQ3_XXS with 4 shards,
Qwen UD-IQ4_XS, and refusals), `expert_layout_test` (the dense layers, the reader pool), `file_expert_source_test`;
`python -m unittest discover -s tools -p "test_iq_pack.py"` (and `test_make_profile.py`, `test_strata_tokenizer.py`)
for the pack, profile and tokenizer; `serve/test_glm.py` and
`serve/chat_golden_glm.json` for the server.

On a GPU (HIP or CUDA), with synthetic data: `dsa_kv_parity`, `glm_conversation_test`
(`-DSTRATA_BUILD_CONVERSATION_TESTS=ON`), `glm_expert_parity` and `glm_pool_parity` (with `STRATA_NATIVE_EXPERTS`).
Against ggml-cpu at llama.cpp f1cee99, registered only when the oracle is built:

```bash
git -C llama.cpp checkout f1cee9941b0e843ea260bf8dd9a090fbd9711b6a
tools/glm/build_oracle.sh llama.cpp build-oracle-shared            # shared ggml-cpu: the glm_* parity tests
cmake -S llama.cpp -B build-oracle-static -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_STATIC=ON -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build-oracle-static --target ggml ggml-cpu ggml-base   # static ggml-cpu: the dsa_* parity tests
cmake -S . -B build ... -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GLM_ORACLE_DIR=llama.cpp -DSTRATA_GLM_ORACLE_BUILD_DIR=build-oracle-shared \
  -DSTRATA_DSA_ORACLE_DIR=llama.cpp -DSTRATA_DSA_ORACLE_BUILD_DIR=build-oracle-static
ctest --test-dir build -R "glm|dsa|geometry|expert_layout"
```

| Test | What, against f1cee99 | Worst error measured (RX 9070 XT) |
|---|---|---|
| `glm_kda_parity` | KDA conv + step, 1-8 tokens, the state after every token; windows vs single steps and `commit(n)` bitwise | 5.5e-7 (output), 2.8e-7 (state) |
| `glm_mhc_parity` | mHC pre (Sinkhorn) / post / mean | 1.9e-7 |
| `glm_router_sigmoid_parity` | sigmoid router with `exp_probs_b`, top-8 of 288 | ids exact, weights 2.6e-7 |
| `glm_ffn_clamp_parity` | dense FFN and shared expert with the SwiGLU clamp | 1.2e-7 vs exact |
| `glm_expert_parity` | routed experts with the clamp; Q6_K, IQ3_S and IQ4_XS down rows; GPU and CPU pool | GPU <= 1.5e-7 |
| `dsa_mla_parity`, `dsa_indexer_parity`, `dsa_kv_parity` | MLA decode stages, the indexer and its selection, the latent cache | selection 198/198 exact; stages <= 9.1e-7 without a Q8_1 code flip |
| `glm_prefill_kda_parity`, `glm_prefill_moe_parity`, `dsa_prompt_parity` | the prompt path's KDA, grouped experts, DSA prompt attention | 3.6e-7; <= 6.2e-4 vs exact; selection exact, 1.8e-6 |

On CUDA `native_mmvq.cu` is built with `--use_fast_math`, so its Q8_1 quantizer rounds quotients within an ulp of a
.5 tie its own way; the DSA tests bound such flips from the GPU's own Q8_1 codes (every code equals the exact rule or
is one step off within 2^-20 of a tie). All of them pass on the 3090 and the 9070 XT.

`tests/glm/serve_e2e.py` drives `serve/server.py` with a real pack over HTTP (streaming, prefix reuse, parking,
thinking off, efforts, a tool-call round trip, cancellation); it needs a GPU and the model.

## Diagnostics

Environment variables of the GLM run (`strata --native <glm5-next shard>` without `--serve`), all off by default:

| Variable | What |
|---|---|
| `STRATA_LOGPOS=<file>` | the teacher-forced top-k rows of every position, the Qwen path's format |
| `STRATA_GLM_LAYERS=N` | run layers 0..N-1 only (no head unless N is the whole model) |
| `STRATA_GLM_DUMP=<dir>`, `_DUMP_LAYERS`, `_DUMP_FROM` | the 4 hyper-connection streams after each layer, llama.cpp's `l_out-<l>` layout, as GLMT files |
| `STRATA_GLM_FROM=i`, `STRATA_GLM_INPUT=<file>` | layers i.. only, on the rows of a GLMT file (llama.cpp's `l_out-(i-1)`) |
| `STRATA_GLM_NODES=<dir>`, `_NODES_LAYER` | one layer's intermediate nodes, named as llama.cpp's eval callback names them |
| `STRATA_GLM_LOGITS=<file>`, `_LOGITS_FROM` | every position's full logits (GLMT) |
| `STRATA_GLM_STATE=<dir>`, `_STATE_AT` | the session after position P: KDA history and state, DSA latents and indexer rows; `tools/glm/glm_check.py state` compares the prompt path's with the windows' |
| `STRATA_GLM_MARK_EVERY=N`, `STRATA_GLM_MARKS=a,b` | a timing line with the tier counters every N tokens / at the listed positions (how the decode segments above were timed) |
| `STRATA_GLM_PROFILE=1`, `STRATA_GLM_PREFILL_TIMING=1` | per-layer GPU times; the prompt path's time per phase |
| `STRATA_GLM_KV=f16` | the DSA latent in f16 |
| `STRATA_GLM_POOL=0\|1`, `_POOL_CALIBRATE=N` | override `--glm-pool`; the calibration's expert count |
| `STRATA_GLM_FETCH=0`, `_LOOKAHEAD=0\|1\|2`, `_LOOKAHEAD_K`, `_PRESTAGE=N`, `_FILE_IO=direct\|pread\|mmap`, `_FETCH_PART`, `_FETCH_SLOTS`, `STRATA_FETCH_THREADS` | the file tier's A/B arms |
| `STRATA_GLM_PREFILL_SHARE=<percent>` | the share of the cache's VRAM `--prefill auto` may give the prompt buffers (default 25) |
| `STRATA_GLM_PREFILL_LOGPOS=L` | `STRATA_LOGPOS` also gets the prompt path's last L positions |
| `STRATA_GLM_PREFILL_SPLIT=a`, `STRATA_GLM_SPLIT_RESTORE=ram` | the prompt read as two calls, with a park and restore between them (the restored rows must be byte-identical) |
| `STRATA_GLM_LOGITS_HASH=1` (`--serve`) | a hash of every decode row's logits in the request's log line |

A GLMT file is a 128-byte header (`GLMT`; u32 version at 4, u32 type at 8 (0 F32, 1 F16, 26 I32), u32 name length
at 12; u64 row elements at 16, u64 rows at 24, i64 ne[4] at 32; the name at 64) followed by the rows. `tools/glm/glm_check.py` reads it
(`lout`, `kld`, `state`).
