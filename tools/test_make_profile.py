"""Tests for tools/make_profile.py, without a model: synthetic profiles and routing traces.

    python -m unittest tools.test_make_profile
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as MP  # noqa: E402

NE = 4   # a tiny "experts per layer"; the layer count is fixed at MP.N_LAYER


def write_trace(path, records):
    """One record per (layer, [expert ids]); the reader skips k weights after the k ids."""
    with open(path, "wb") as f:
        for layer, experts in records:
            f.write(struct.pack("<ii", layer, len(experts)))
            f.write(struct.pack("<%di" % len(experts), *experts))
            f.write(struct.pack("<%df" % len(experts), *([1.0] * len(experts))))


def full_base():
    """Every pair, in an order that differs from rank_profile's fill (e-major then layer)."""
    return [(layer, e) for layer in range(MP.N_LAYER) for e in range(NE)]


class RankProfile(unittest.TestCase):
    def test_a_complete_base_hides_the_trace(self):
        # the shipped profile's situation: it already ranks every pair, so the trace cannot move anything
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked, base)
        self.assertEqual(counts["trace"], 0)
        self.assertEqual(counts["base"], MP.N_LAYER * NE)

    def test_reorder_puts_the_traces_first(self):
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE, reorder=True)
        self.assertEqual(ranked[:2], [(5, 1), (6, 2)])
        self.assertEqual(set(ranked), set(base))                 # still every pair, once
        self.assertEqual(len(ranked), len(base))
        self.assertEqual(counts["trace"], 2)
        self.assertEqual(counts["base"], MP.N_LAYER * NE - 2)    # the base fills the rest, minus the two

    def test_frequency_and_tie_break_order_the_trace(self):
        freq = defaultdict(int, {(1, 0): 1, (2, 0): 3, (3, 0): 3})
        ranked, _ = MP.rank_profile([], freq, NE, no_base=True)  # (2,0)/(3,0) tie -> the smaller pair first
        self.assertEqual(ranked[:3], [(2, 0), (3, 0), (1, 0)])

    def test_default_appends_only(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(0, 0): 5, (1, 3): 9})
        ranked, _ = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked[:3], [(0, 0), (0, 1), (1, 3)])   # base first; (0,0) already there, not moved

    def test_no_base_uses_the_traces_then_the_fill(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(1, 3): 9, (0, 0): 5})
        ranked, counts = MP.rank_profile(base, freq, NE, no_base=True)
        self.assertEqual(ranked[:2], [(1, 3), (0, 0)])
        self.assertEqual(counts["base"], 0)
        self.assertEqual(len(ranked), MP.N_LAYER * NE)

    def test_reorder_without_traces_is_the_default(self):
        base = full_base()
        self.assertEqual(MP.rank_profile(base, defaultdict(int), NE, reorder=True)[0], base)


class Main(unittest.TestCase):
    def run_main(self, argv):
        old, sys.argv = sys.argv, ["make_profile.py"] + argv
        try:
            MP.main()
        finally:
            sys.argv = old

    def test_end_to_end_reorder_over_a_complete_base(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.bin"
            out = Path(d) / "out.bin"
            trace = Path(d) / "t.bin"
            MP.write_profile(str(base), full_base(), NE)
            write_trace(str(trace), [(5, [1]), (6, [2]), (6, [2]), (6, [2])])
            self.run_main([str(trace), "--base", str(base), "--reorder", "--n-expert", str(NE), "--out", str(out)])
            ranked = MP.read_profile(str(out), NE)
            self.assertEqual(ranked[:2], [(6, 2), (5, 1)])       # most frequent first
            self.assertEqual(set(ranked), set(full_base()))
            # and without --reorder it is the base's order, untouched
            self.run_main([str(trace), "--base", str(base), "--n-expert", str(NE), "--out", str(out)])
            self.assertEqual(MP.read_profile(str(out), NE), full_base())



class Geometry(unittest.TestCase):
    """GLM-5.3-Flash's shape in small: layers 0-2 without experts, 3-5 with 4, a 7th block for MTP."""

    def pack(self, d):
        lines = ["# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
                 "(n_expert 4, total 48; absolute offsets in m.gguf, or in the named shard beside it)"]
        lines += ["%d 21 23 %d 4 0 0 0" % (l, 16 * (l - 3)) for l in (3, 4, 5)]
        (Path(d) / "native_experts.txt").write_text("\n".join(lines) + "\n")
        return d

    def test_from_pack(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(MP.geometry_from_pack(self.pack(d)), (6, 4, [3, 4, 5]))

    def test_from_gguf(self):
        from _paths import add_gguf_py
        add_gguf_py()
        from gguf import GGUFWriter
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.gguf"
            w = GGUFWriter(path, "glm5-next")
            for k, v in (("block_count", 7), ("nextn_predict_layers", 1), ("expert_count", 4),
                         ("leading_dense_block_count", 3)):
                w.add_uint32("glm5-next." + k, v)
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.close()
            self.assertEqual(MP.geometry_from_gguf(path), (6, 4, [3, 4, 5]))

    def test_layers_without_experts_are_never_ranked(self):
        with tempfile.TemporaryDirectory() as d:
            trace, out = Path(d) / "t.bin", Path(d) / "out.bin"
            # layer 1 has no experts (a llama.cpp trace never names it; a wrong one is skipped), layer 4 does
            write_trace(str(trace), [(1, [0, 1]), (4, [2]), (4, [2]), (5, [3])])
            freq = MP.read_trace(str(trace), 4, 6, [3, 4, 5])
            self.assertEqual(dict(freq), {(4, 2): 2, (5, 3): 1})
            ranked, counts = MP.rank_profile([], freq, 4, no_base=True, n_layer=6, layers=[3, 4, 5])
            self.assertEqual(ranked[:2], [(4, 2), (5, 3)])
            self.assertEqual(len(ranked), 12)                         # 3 layers x 4, not 6 x 4
            self.assertFalse([p for p in ranked if p[0] < 3])
            MP.write_profile(str(out), ranked, 4, 6)
            blob = out.read_bytes()
            self.assertEqual(struct.unpack_from("<4s5I", blob), (b"STRP", 1, 6, 4, 12, 12))
            self.assertEqual(MP.read_profile(str(out), 4, 6, [3, 4, 5]), ranked)
            with self.assertRaises(SystemExit):                        # a base that ranks a dense layer
                MP.read_profile(str(out), 4, 6, [4, 5])

    def test_main_with_a_pack(self):
        with tempfile.TemporaryDirectory() as d:
            trace, out = Path(d) / "t.bin", Path(d) / "out.bin"
            write_trace(str(trace), [(3, [1]), (3, [1]), (5, [0])])
            old, sys.argv = sys.argv, ["make_profile.py", str(trace), "--pack", self.pack(d), "--no-base",
                                       "--out", str(out)]
            try:
                MP.main()
            finally:
                sys.argv = old
            self.assertEqual(MP.read_profile(str(out), 4, 6)[:2], [(3, 1), (5, 0)])
            self.assertEqual(len(MP.read_profile(str(out), 4, 6)), 12)

if __name__ == "__main__":
    unittest.main()
