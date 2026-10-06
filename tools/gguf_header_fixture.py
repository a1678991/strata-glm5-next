#!/usr/bin/env python3
"""Header-only GGUF fixtures of real models, for the loader and geometry tests (no weights, no numpy, no GPU).

A header-only GGUF is a real file's metadata and tensor directory - every key, every tensor name, shape, type and
offset - followed by nothing: the data section is empty.  Strata's reader (include/strata/artifact/gguf_reader.hpp)
parses it like the real file; anything that would read a payload (`GgufModel::in_bounds`, the native loaders'
truncation checks) sees it as truncated, which the tests use as the proof that every check BEFORE the payload passed.

Two steps, both deterministic:

  trim   a header dump (research/gguf/gguf_header.py's JSON: one object per shard with `kv` and `tensors`) ->
         a compact spec committed under tests/data/gguf_headers/.  The dump keeps arrays longer than 64 entries only
         as {"array_of", "len", "head"}; the spec keeps their length and first values.  It does not keep the value
         types (the dump has Python values), so `write` chooses them the way gguf-py does: str STRING, bool BOOL,
         float FLOAT32, int UINT32 (INT32 if negative, UINT64 above 32 bits); arrays of int INT32 (UINT64 if a value
         needs it), of float FLOAT32, of str STRING.  split.no / split.count are UINT16 and split.tensors.count
         INT32, as llama.cpp's gguf-split writes them.
  write  spec -> one header-only .gguf per shard, under the shard's own file name, in an output directory.
         A shortened array is written at its full length: the first values as recorded, then zeros (numbers) or
         empty strings - the reader counts them, nothing here reads the vocabulary's strings.

  python tools/gguf_header_fixture.py trim  DUMP.json SPEC.json
  python tools/gguf_header_fixture.py write SPEC.json OUT_DIR
"""
from __future__ import annotations

import json
import os
import struct
import sys

GGML_TYPES = {"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8, "Q8_1": 9, "Q2_K": 10,
              "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "Q8_K": 15, "IQ2_XXS": 16, "IQ2_XS": 17,
              "IQ3_XXS": 18, "IQ1_S": 19, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23, "I8": 24,
              "I16": 25, "I32": 26, "I64": 27, "F64": 28, "IQ1_M": 29, "BF16": 30, "TQ1_0": 34, "TQ2_0": 35,
              "MXFP4": 39, "Q2_0": 42}

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
SCALAR = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}
FORCED = {"split.no": U16, "split.count": U16, "split.tensors.count": I32}
ALIGN = 32


def trim(dump_path: str, spec_path: str) -> None:
    shards = json.load(open(dump_path, encoding="utf-8"))
    out = {"source": os.path.basename(dump_path), "shards": []}
    for s in shards:
        kv = []
        for k, v in s["kv"].items():
            if isinstance(v, dict):   # a long array the dump summarised
                kv.append([k, {"len": v["len"], "of": v["array_of"], "head": v.get("head") or []}])
            else:
                kv.append([k, v])
        tensors = [[t["name"], t["dims"], t["type"], t["offset"]] for t in s["tensors"]]
        out["shards"].append({"file": os.path.basename(s["file"]), "kv": kv, "tensors": tensors})
    with open(spec_path, "w", encoding="utf-8", newline="\n") as f:
        # one key or tensor per line: diffable, and small enough to commit
        f.write('{"source": %s, "shards": [\n' % json.dumps(out["source"]))
        for i, s in enumerate(out["shards"]):
            f.write('{"file": %s,\n "kv": [\n' % json.dumps(s["file"]))
            f.write(",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in s["kv"]))
            f.write('\n ],\n "tensors": [\n')
            f.write(",\n".join("  " + json.dumps(t) for t in s["tensors"]))
            f.write("\n ]}%s\n" % ("," if i + 1 < len(out["shards"]) else ""))
        f.write("]}\n")


def _int_type(vals) -> int:
    if all(-2**31 <= x < 2**31 for x in vals):
        return I32
    return U64 if all(x >= 0 for x in vals) else I64


def _scalar(key: str, v):
    if key in FORCED:
        return FORCED[key], v
    if isinstance(v, bool):
        return BOOL, v
    if isinstance(v, int):
        if v < 0:
            return I32, v
        return (U32 if v < 2**32 else U64), v
    if isinstance(v, float):
        return F32, v
    if isinstance(v, str):
        return STRING, v
    raise ValueError(f"{key}: unsupported value {v!r}")


def _put_str(b: bytearray, s: str) -> None:
    raw = s.encode("utf-8")
    b += struct.pack("<Q", len(raw))
    b += raw


def _put_kv(b: bytearray, key: str, v) -> None:
    _put_str(b, key)
    if isinstance(v, dict):   # shortened array: {"len", "of", "head"}
        n, head = v["len"], v["head"]
        if v["of"] == "string":
            b += struct.pack("<IIQ", ARRAY, STRING, n)
            for i in range(n):
                _put_str(b, head[i] if i < len(head) else "")
            return
        code = {"B": U8, "b": I8, "H": U16, "h": I16, "I": U32, "i": I32, "f": F32, "?": BOOL, "Q": U64, "q": I64,
                "d": F64}[v["of"]]
        b += struct.pack("<IIQ", ARRAY, code, n)
        vals = list(head) + [0] * (n - len(head))
        b += struct.pack("<%d%s" % (n, SCALAR[code]), *vals)
        return
    if isinstance(v, list):
        if all(isinstance(x, str) for x in v):
            b += struct.pack("<IIQ", ARRAY, STRING, len(v))
            for x in v:
                _put_str(b, x)
        elif all(isinstance(x, bool) for x in v):
            b += struct.pack("<IIQ%d?" % len(v), ARRAY, BOOL, len(v), *v)
        elif all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            t = _int_type(v)
            b += struct.pack("<IIQ%d%s" % (len(v), SCALAR[t]), ARRAY, t, len(v), *v)
        elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            b += struct.pack("<IIQ%df" % len(v), ARRAY, F32, len(v), *[float(x) for x in v])
        else:
            raise ValueError(f"{key}: mixed array")
        return
    t, x = _scalar(key, v)
    b += struct.pack("<I", t)
    if t == STRING:
        _put_str(b, x)
    else:
        b += struct.pack("<" + SCALAR[t], x)


def load_spec(spec_path: str) -> dict:
    return json.load(open(spec_path, encoding="utf-8"))


def write(spec_path: str, out_dir: str) -> list[str]:
    return write_spec(load_spec(spec_path), out_dir)


def write_spec(spec: dict, out_dir: str) -> list[str]:
    """Writes every shard of `spec` (as `load_spec` returns it, possibly changed by the caller) into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for s in spec["shards"]:
        b = bytearray(b"GGUF")
        b += struct.pack("<IQQ", 3, len(s["tensors"]), len(s["kv"]))
        for key, v in s["kv"]:
            _put_kv(b, key, v)
        for name, dims, tname, offset in s["tensors"]:
            _put_str(b, name)
            b += struct.pack("<I%dQ" % len(dims), len(dims), *dims)
            b += struct.pack("<IQ", GGML_TYPES[tname], offset)
        b += b"\0" * ((-len(b)) % ALIGN)   # the data section starts here and holds nothing
        path = os.path.join(out_dir, s["file"])
        with open(path, "wb") as f:
            f.write(b)
        paths.append(path)
    return paths


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "trim":
        trim(argv[2], argv[3])
        return 0
    if len(argv) == 4 and argv[1] == "write":
        for p in write(argv[2], argv[3]):
            print(p)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
