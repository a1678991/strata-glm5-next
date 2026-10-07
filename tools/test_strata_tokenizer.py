"""Tests for tools/strata_tokenizer.py's pre-tokenizer types, without a model: a toy byte-level vocabulary.

    python -m unittest discover -s tools -p test_strata_tokenizer.py

The real check is against llama.cpp: `llama-tokenize` (f1cee99) on GLM-5.3-Flash's vocabulary gave the same ids as
this tokenizer on 117 of 117 texts (47,747 ids; docs/GLM.md); these tests pin the mechanics.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

B = ST.BYTE_TO_UNICODE


def toy(pre, extra=(), merges=()):
    """Every byte as a token, then `extra` tokens (byte-mapped strings), with `merges` (pairs of byte-mapped)."""
    tokens = [B[b] for b in range(256)] + list(extra) + ["<|user|>", "<think>"]
    types = [1] * (len(tokens) - 2) + [3, 4]
    return ST.Tokenizer(tokens, [" ".join(m) for m in merges], types, pre)


class PreTypes(unittest.TestCase):
    def test_unknown_pre_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not transcribed"):
            toy("llama3")

    def test_digits_in_threes_for_glm4_one_by_one_for_qwen35(self):
        g, q = toy("glm4"), toy("qwen35")
        self.assertEqual(g._re.findall("1234567 x"), ["123", "456", "7", " x"])
        self.assertEqual(q._re.findall("1234"), ["1", "2", "3", "4"])

    def test_combining_marks_split_for_glm4_not_for_qwen35(self):
        # glm4 is llama3's expression: \p{L} without \p{M}, so a combining mark starts a punctuation piece
        self.assertEqual(toy("glm4")._re.findall("éx"), ["e", "́x"])
        self.assertEqual(toy("qwen35")._re.findall("éx"), ["éx"])

    def test_ignore_merges_takes_a_whole_piece_from_the_vocabulary(self):
        # "abc" is a token but no merge builds it: glm4 emits it whole (llama.cpp's ignore_merges), qwen35 does not
        abc = B[ord("a")] + B[ord("b")] + B[ord("c")]
        g = toy("glm4", extra=[abc])
        q = toy("qwen35", extra=[abc])
        self.assertTrue(g.ignore_merges)
        self.assertFalse(q.ignore_merges)
        self.assertEqual(g.encode("abc"), [256])
        self.assertEqual(q.encode("abc"), [ord("a"), ord("b"), ord("c")])
        self.assertEqual(g.decode(g.encode("abc abd")), "abc abd")

    def test_control_and_user_defined_literals(self):
        g = toy("glm4")
        user, think = len(g.tokens) - 2, len(g.tokens) - 1
        self.assertEqual(g.encode("<think>x"), [think, ord("x")])               # type 4: always
        self.assertNotIn(user, g.encode("<|user|>"))                            # type 3: only with parse_special
        self.assertEqual(g.encode("<|user|>", parse_special=True), [user])

    def test_tokenizer_dir_records_the_pre_type(self):
        with tempfile.TemporaryDirectory() as d:
            for pre in ("glm4", "qwen35"):
                tk = toy(pre)
                out = Path(d) / pre / "tokenizer"
                out.mkdir(parents=True)
                (out / "vocab.json").write_text(json.dumps(tk.ids), encoding="utf-8")
                (out / "merges.txt").write_text("", encoding="utf-8")
                (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
                (out / "tokenizer.json").write_text(json.dumps({"pre": pre}), encoding="utf-8")
                back = ST.Tokenizer.from_dir(out)
                self.assertEqual((back.pre, back.ignore_merges), (pre, pre == "glm4"))
            (out / "tokenizer.json").unlink()                                   # a pack from before GLM
            self.assertEqual(ST.Tokenizer.from_dir(out).pre, "qwen35")


if __name__ == "__main__":
    unittest.main()
