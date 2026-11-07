"""Load the shared ALBench reservoir pools and evaluation sets. numpy only, no install needed.

Copy this next to your own code, or `sys.path.insert(0, "/grid/koo/home/shared/albench/reservoir_pools")`.
It exists so the three things that are easy to get wrong are hard to get wrong: pools are PREFIX-nested
(never reshuffle them), students see the insert inside 20 real plasmid bases per side, and predictions are
averaged over both strands.

    import albench_pools as ap
    seqs, y = ap.load_pool("random", 30_000)      # the study's 30k random pool, oracle_v3 labels
    x = ap.wrap(seqs)                             # 240-bp model input
    ev_seqs, ev_y = ap.load_eval("genomic")       # held-out chr7/13, the headline test set
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SHARE = Path("/grid/koo/home/shared/albench")
POOLS = SHARE / "reservoir_pools"
EVALS = SHARE / "data/eval_sets"
FLANK_CTX = 20  # the study's standard: 20 real plasmid bases each side, 240-bp input
SHIFT_MAX = 10  # training-time window shift, taken from real plasmid bases (never zeros)


def _flanks() -> tuple[str, str]:
    up = dn = ""
    for line in (SHARE / "data/plasmid_flank.txt").read_text().splitlines():
        if line.startswith("upstream_300"):
            up = line.split()[1]
        elif line.startswith("downstream_300"):
            dn = line.split()[1]
    if not up or not dn:
        raise RuntimeError("plasmid_flank.txt did not contain both flanks")
    return up, dn


def list_pools() -> dict:
    """Every published reservoir, with n, labels, GC and provenance."""
    return json.loads((POOLS / "MANIFEST.json").read_text())["reservoir_pools"]


def list_evals() -> dict:
    return json.loads((EVALS / "MANIFEST.json").read_text())["eval_sets"]


def _open(path: Path, n: int | None):
    z = np.load(path, allow_pickle=True)
    skey = next(k for k in ("sequences", "ref_sequences") if k in z.files)
    ykey = next(k for k in ("oracle_labels", "oracle_mean", "ref_mean") if k in z.files)
    seqs = [str(s) for s in z[skey][:n]]
    return seqs, z[ykey][:n].astype(np.float32)


def load_pool(strategy: str, n: int | None = None) -> tuple[list[str], np.ndarray]:
    """The first ``n`` sequences of a reservoir's canonical pool, with oracle_v3 labels.

    Every size in the study is a PREFIX of this file, so ``n`` gives exactly the pool the study's
    n-sequence results were produced on. Do not shuffle before slicing: that turns a size comparison
    into a comparison of different samples.
    """
    hits = sorted(POOLS.glob(f"{strategy}_d*.npz"))
    if not hits:
        raise FileNotFoundError(f"{strategy}: not published; have {sorted(list_pools())}")
    seqs, y = _open(hits[-1], n)
    if n is not None and len(seqs) < n:
        raise ValueError(f"{strategy} holds {len(seqs)} sequences, fewer than the {n} requested")
    return seqs, y


def load_eval(name: str, n: int | None = None) -> tuple[list[str], np.ndarray]:
    """An evaluation set by short name, e.g. ``genomic``, ``ood``, ``random_32k``, ``sub_high``."""
    p = EVALS / f"{name}_oracle.npz"
    if not p.exists():
        raise FileNotFoundError(f"{name}: have {sorted(e['set'] for e in list_evals().values())}")
    return _open(p, n)


def load_val(n: int | None = None) -> tuple[list[str], np.ndarray]:
    """V0c, the validation set. Fit ensemble weights and early-stop on it; NEVER train on it."""
    return _open(SHARE / "data/val_V0c.npz", n)


def wrap(seqs, ctx: int = FLANK_CTX, margin: int = 0) -> list[str]:
    """``upstream[-(ctx+margin):] + insert + downstream[:ctx+margin]`` -- the study's model input.

    ``margin`` > 0 returns a wider window so training can shift by up to ``margin`` real bases; crop
    the centre at evaluation.
    """
    up, dn = _flanks()
    k = ctx + margin
    if k == 0:
        return [str(s) for s in seqs]
    return [up[-k:] + str(s) + dn[:k] for s in seqs]


def revcomp(seqs) -> list[str]:
    """Both-strand averaging at evaluation is part of the standard: predict on these too and mean."""
    table = str.maketrans("ACGTN", "TGCAN")
    return [str(s).translate(table)[::-1] for s in seqs]


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    pools = list_pools()
    print(f"{len(pools)} reservoir pools in {POOLS}")
    for name, info in sorted(pools.items(), key=lambda kv: (kv[1]["group"], kv[0])):
        sizes = "/".join(f"{s // 1000}k" for s in info["prefix_sizes"])
        print(f"  {info['strategy']:20s} {info['n']:>7,}  prefixes {sizes:16s} {info['doc']}")
    print(f"\n{len(list_evals())} evaluation sets in {EVALS}")
