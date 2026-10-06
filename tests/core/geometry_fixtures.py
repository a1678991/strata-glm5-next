#!/usr/bin/env python3
"""Writes the header-only GGUF fixtures of model_geometry_test into OUT_DIR (tools/gguf_header_fixture.py).

  glm/, qwen/     GLM-5.3-Flash UD-IQ4_XS and Qwen3.8-Flash-Next UD-IQ4_XS, metadata and tensor directories as read
                  on 2026-10-06 (tests/data/gguf_headers/*.json), no tensor data
  glm-*/          one change each, which the guard, the geometry or the tensor check must refuse
  llama/          the Qwen file under another architecture name

Usage: python tests/core/geometry_fixtures.py OUT_DIR
"""
import copy
import os
import sys

sys.dont_write_bytecode = True   # nothing written into the source tree at test time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gguf_header_fixture as fx  # noqa: E402

DATA = os.path.join(ROOT, "tests", "data", "gguf_headers")


def kv_set(spec, key, fn):
    for e in spec["shards"][0]["kv"]:
        if e[0] == key:
            e[1] = fn(e[1])
            return spec
    raise KeyError(key)


def tensor_set(spec, name, dims):
    for s in spec["shards"]:
        for t in s["tensors"]:
            if t[0] == name:
                t[1] = dims
                return spec
    raise KeyError(name)


def with_list(i, value):
    def f(v):
        v = list(v)
        v[i] = value
        return v
    return f


def main(out):
    glm = fx.load_spec(os.path.join(DATA, "glm5-next-ud-iq4_xs.json"))
    qwen = fx.load_spec(os.path.join(DATA, "qwen4exp-ud-iq4_xs.json"))
    variants = {
        "glm": glm,
        "qwen": qwen,
        # a layer type the kernels have no path for (MLA with 2 latents)
        "glm-kv2": kv_set(copy.deepcopy(glm), "glm5-next.attention.head_count_kv", with_list(5, 2)),
        # a pinned shape key
        "glm-embd": kv_set(copy.deepcopy(glm), "glm5-next.embedding_length", lambda v: 4097),
        # one tensor of the wrong shape
        "glm-shape": tensor_set(copy.deepcopy(glm), "blk.4.attn_q.weight", [4096, 8191]),
        # layer 4 declared DSA while its tensors are KDA's: the per-layer kinds disagree with the tensors
        "glm-swap": kv_set(copy.deepcopy(glm), "glm5-next.attention.head_count_kv", with_list(4, 1)),
        # a per-layer value the kernels take once per model
        "glm-clamp": kv_set(copy.deepcopy(glm), "glm5-next.swiglu_clamp_exp", with_list(7, 9.0)),
        "llama": kv_set(copy.deepcopy(qwen), "general.architecture", lambda v: "llama"),
    }
    for name, spec in variants.items():
        fx.write_spec(spec, os.path.join(out, name))
    print("wrote %d fixtures into %s" % (len(variants), out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
