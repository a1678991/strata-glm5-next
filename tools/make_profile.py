"""tools/make_profile.py - the expert-cache profile (`data/expert-profile.bin`): every (layer, expert) pair, ranked.

`--expert-cache auto` fills the VRAM slots it can afford with the profile's pairs in order, so the ranking decides
which experts start resident (the adaptive tier then swaps in what a conversation routes most).  A profile that
ranks fewer pairs than a card can hold caps the cache (issue #46: a 32 GB card stopped at 8,000 slots); this tool
writes one that ranks all 24,576.

The order: the base profile's ranking (default: the shipped data/expert-profile.bin), then the pairs your routing
traces used, most frequent first, then every pair still missing, interleaved across the layers.  The base is only
*appended to*, never reordered - and the shipped profile already ranks all 24,576 pairs, so with it a trace is a
no-op.  `--reorder` ranks the traces first and lets the base fill the rest, which is how a workload's trace decides
the top; `--no-base` drops the base entirely.

WATCH THE BASE: `take()` skips a pair it has already ranked, and the shipped base ranks all 24,576, so a plain
`--base` run cannot be moved by any trace - the output is the base again.  `--reorder` (traces first) or
`--no-base` is what makes a trace decide the top.  Which ordering is in use is not cosmetic: the layer-split cost
model reads this RANKING as if it were a frequency curve, and the coverage curve in src/program/generate.cpp
carries the measured hit rates that say how far off that goes (94.9% claimed against 69.2% measured at 5,805 pairs
held, on the IQ3_S 4-way rig).

    python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--reorder] [--out PATH]
                                 [--n-expert 256]      (a pruned model: GSQ-RCO Coder keeps 256 of 512)
                                 [--pack DIR | --gguf SHARD1]   (the dimensions from the model instead of 48 x 512)

A routing trace comes from a one-shot engine run with `--dump-routing FILE` (a prompt typical of your use; the
routed experts of every layer and position are written).  Point the model config's `--expert-profile` at the result.

The dimensions.  The profile's header holds (layers, experts per layer), which the engine checks against the model
(src/core/expert_cache.cpp, read_expert_profile).  Without --pack/--gguf they are Qwen3.8-Flash-Next's 48 x 512
(--n-expert overrides the experts).  --pack reads them from the pack's native_experts.txt - n_expert from its
header, the layers from its lines, the layer count = the last one + 1 - which is what the engine loads, so it is
the one to prefer.  --gguf reads the GGUF's metadata instead: `<arch>.block_count` minus `nextn_predict_layers`
(the MTP block is not a trunk layer), `<arch>.expert_count`, and `leading_dense_block_count` (glm5-next: 3).
A layer without experts (GLM-5.3-Flash's blk.0-2) keeps its row in the header's layer count but no pair of it is
ranked, filled in or taken from a trace or a base, so GLM's profile is 45 x 288 with 42 x 288 = 12,096 pairs.  A
GLM trace comes from llama.cpp (an eval-callback dump of ffn_moe_topk-N per token) in the same record format.
"""
import argparse
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512          # Qwen3.8-Flash-Next: the defaults without --pack / --gguf
MAGIC, VERSION = b"STRP", 1


def geometry_from_pack(pack):
    """(n_layer, n_expert, MoE layers) from a pack's native_experts.txt (tools/iq_pack.py)."""
    text = (Path(pack) / "native_experts.txt").read_text(encoding="utf-8")
    m = re.search(r"\(n_expert (\d+),", text)
    layers = [int(line.split()[0]) for line in text.splitlines() if line and not line.startswith("#")]
    if m is None or not layers:
        raise SystemExit(f"{pack}/native_experts.txt: no n_expert header or no layer lines")
    return max(layers) + 1, int(m.group(1)), sorted(layers)


def geometry_from_gguf(shard1):
    """(n_layer, n_expert, MoE layers) from a GGUF's metadata (shard 1 carries it)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    md = GGUFFile(Path(shard1)).metadata
    arch = md.get("general.architecture")
    try:
        n_layer = int(md[f"{arch}.block_count"]) - int(md.get(f"{arch}.nextn_predict_layers", 0))
        n_expert = int(md[f"{arch}.expert_count"])
    except KeyError as e:
        raise SystemExit(f"{shard1}: no {e.args[0]} in the metadata")
    lead = int(md.get(f"{arch}.leading_dense_block_count", 0))
    return n_layer, n_expert, list(range(lead, n_layer))


def read_profile(path, n_expert=N_EXPERT, n_layer=None, layers=None):
    n_layer = N_LAYER if n_layer is None else n_layer   # at call time: callers may set N_LAYER
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (nl, ne) != (n_layer, n_expert):
        raise SystemExit(f"{path}: {nl}x{ne}, not {n_layer}x{n_expert} (a profile of another model: "
                         "--no-base, or a --base of this one)")
    pairs = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]
    if layers is not None:
        moe = set(layers)
        bad = sorted({layer for layer, _ in pairs if layer not in moe})
        if bad:
            raise SystemExit(f"{path}: ranks experts of layers {bad}, which have none")
    return pairs


def read_trace(path, n_expert=N_EXPERT, n_layer=None, layers=None):
    """(layer, k, k expert ids, k weights) records, as `--dump-routing` writes them.  Records of a layer outside
    `layers` (one without experts) or of an id beyond n_expert are skipped, as out-of-range ones always were."""
    n_layer = N_LAYER if n_layer is None else n_layer   # at call time: callers may set N_LAYER
    moe = set(range(n_layer) if layers is None else layers)
    blob = Path(path).read_bytes()
    off, freq = 0, defaultdict(int)
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        for e in struct.unpack_from("<%di" % k, blob, off):
            if layer in moe and 0 <= e < n_expert:
                freq[(layer, e)] += 1
        off += 8 * k                                    # the ids and the weights
    return freq


def write_profile(path, ranked, n_expert=N_EXPERT, n_layer=None):
    n_layer = N_LAYER if n_layer is None else n_layer   # at call time: callers may set N_LAYER
    table = [[-1] * n_expert for _ in range(n_layer)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layer, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layer):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def rank_profile(base_pairs, trace_freq, n_expert=N_EXPERT, no_base=False, reorder=False, n_layer=None,
                 layers=None):
    """The ranked (layer, expert) list and the counts it was built from (`base`, `trace`, `fill`).

    Default: the base's order, then the traces' pairs most frequent first, then the fill.  With `reorder` (or
    `no_base`) the traces' pairs come first, so a base that already ranks every pair no longer hides them; the base
    (unless `no_base`) and then the fill follow.  A base is never reordered against itself - it only ever fills."""
    n_layer = N_LAYER if n_layer is None else n_layer   # at call time: callers may set N_LAYER
    ranked, seen, counts = [], set(), defaultdict(int)

    def take(pairs, key):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)
                counts[key] += 1

    trace_pairs = [p for p, _ in sorted(trace_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    if no_base or reorder:
        take(trace_pairs, "trace")
    if not no_base:
        take(base_pairs, "base")
    if not (no_base or reorder):
        take(trace_pairs, "trace")
    moe = list(range(n_layer)) if layers is None else sorted(layers)
    take(((layer, e) for e in range(n_expert) for layer in moe), "fill")
    return ranked, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="routing traces from --dump-routing")
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"), help="ranking to keep first")
    ap.add_argument("--no-base", action="store_true", help="rank by the traces only")
    ap.add_argument("--reorder", action="store_true",
                    help="rank the traces' pairs before the base's (the base, then the fill, follow); needed with a "
                         "base that already ranks every pair, where --base alone cannot change the order")
    ap.add_argument("--out", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--n-expert", type=int, default=None, help="experts per layer (default 512, or the model's)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--pack", help="take the dimensions from this pack's native_experts.txt (preferred)")
    src.add_argument("--gguf", help="take the dimensions from this GGUF's metadata (shard 1)")
    a = ap.parse_args()

    if a.pack or a.gguf:
        nl, ne, layers = geometry_from_pack(a.pack) if a.pack else geometry_from_gguf(a.gguf)
        if a.n_expert is not None and a.n_expert != ne:
            ap.error(f"--n-expert {a.n_expert}, but the model has {ne}")
    else:
        nl, ne, layers = N_LAYER, N_EXPERT if a.n_expert is None else a.n_expert, None
    freq = defaultdict(int)
    for t in a.traces:
        for p, c in read_trace(t, ne, nl, layers).items():
            freq[p] += c
    base_pairs = [] if a.no_base else read_profile(a.base, ne, nl, layers)
    ranked, counts = rank_profile(base_pairs, freq, ne, no_base=a.no_base, reorder=a.reorder, n_layer=nl,
                                  layers=layers)
    write_profile(a.out, ranked, ne, nl)
    assert read_profile(a.out, ne, nl, layers) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {nl} x {ne}, {len(ranked)} ranked pairs ({counts['base']} from the base, "
          f"{counts['trace']} from the traces, {counts['fill']} filled in)")


if __name__ == "__main__":
    main()
