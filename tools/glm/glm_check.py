#!/usr/bin/env python3
"""tools/glm/glm_check.py - the GLM port's numeric checks on GLMT files (validate/scripts/glmt.py's format).

    glm_check.py lout OURS REF [--json OUT]
        Per-token relative error of a layer output (l_out-N: 4 hc streams x n_embd per row) against llama.cpp's dump:
        rel_l2[t] = |ours[t] - ref[t]|_2 / |ref[t]|_2 and rel_max[t] = max|ours[t] - ref[t]| / max|ref[t]|.
        Prints mean / median / p99 / max of both over the tokens, and the token with the largest rel_l2.
    glm_check.py kld REF_LOGITS OURS_LOGITS [--from P] [--json OUT]
        Full-vocabulary KL(REF || OURS) per position from two `glm-splice --logits` files (every vocabulary entry, in
        float64), its mean, median, p99 and max, and the share of positions with the same argmax ("same top p") -
        llama-perplexity --kl-divergence's two headline figures, over every position of one teacher-forced pass.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

TYPES = {0: np.float32, 26: np.int32, 1: np.float16}


def open_glmt(path):
    with open(path, "rb") as f:
        h = f.read(128)
    if h[:4] != b"GLMT":
        raise SystemExit(f"{path}: not a GLMT file")
    _, ty, nl = struct.unpack_from("<3I", h, 4)
    row_elems, rows = struct.unpack_from("<2Q", h, 16)
    name = h[64:64 + nl].decode()
    data = np.memmap(path, dtype=TYPES[ty], mode="r", offset=128, shape=(rows, row_elems))
    return name, data


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "n": int(v.size)}


def cmd_lout(a):
    na, A = open_glmt(a.ours)
    nb, B = open_glmt(a.ref)
    if A.shape != B.shape:
        raise SystemExit(f"shapes differ: {A.shape} vs {B.shape}")
    l2, mx = [], []
    for t in range(A.shape[0]):
        x = A[t].astype(np.float64)
        y = B[t].astype(np.float64)
        d = x - y
        l2.append(np.linalg.norm(d) / max(np.linalg.norm(y), 1e-30))
        mx.append(np.abs(d).max() / max(np.abs(y).max(), 1e-30))
    r = {"ours": na, "ref": nb, "rows": int(A.shape[0]), "rel_l2": stats(l2), "rel_max": stats(mx),
         "worst_token": int(np.argmax(l2))}
    print(f"{na} vs {nb}: {r['rows']} tokens; rel L2 mean {r['rel_l2']['mean']:.3e} median {r['rel_l2']['median']:.3e} "
          f"p99 {r['rel_l2']['p99']:.3e} max {r['rel_l2']['max']:.3e} (token {r['worst_token']}); rel max-abs mean "
          f"{r['rel_max']['mean']:.3e} max {r['rel_max']['max']:.3e}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0


def cmd_kld(a):
    _, R = open_glmt(a.ref)
    _, O = open_glmt(a.ours)
    if R.shape != O.shape:
        raise SystemExit(f"shapes differ: {R.shape} vs {O.shape}")
    kl, same = [], 0
    lo = a.from_pos
    for t in range(lo, R.shape[0]):
        r = R[t].astype(np.float64)
        o = O[t].astype(np.float64)
        lr = r - (r.max() + np.log(np.exp(r - r.max()).sum()))
        lo_ = o - (o.max() + np.log(np.exp(o - o.max()).sum()))
        kl.append(float((np.exp(lr) * (lr - lo_)).sum()))
        same += int(np.argmax(r) == np.argmax(o))
    n = len(kl)
    s = stats(kl)
    se = float(np.std(kl, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    res = {"positions": n, "kld": s, "kld_se": se, "same_top": same / n}
    print(f"{n} positions: full-vocab KLD mean {s['mean']:.5f} +- {se:.5f}, median {s['median']:.5f}, p99 "
          f"{s['p99']:.4f}, max {s['max']:.4f}; same top token {100.0 * same / n:.2f}%")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lout")
    p.add_argument("ours")
    p.add_argument("ref")
    p.add_argument("--json")
    p = sub.add_parser("kld")
    p.add_argument("ref")
    p.add_argument("ours")
    p.add_argument("--from", dest="from_pos", type=int, default=0)
    p.add_argument("--json")
    a = ap.parse_args(argv)
    return cmd_lout(a) if a.cmd == "lout" else cmd_kld(a)


if __name__ == "__main__":
    sys.exit(main())
