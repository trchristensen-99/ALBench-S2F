"""Label a collaborator reservoir with oracle_v3 and write its nested canonical pools (10k, 30k, 100k) into k562_nested/.

The subsets are the collaborator's own: `albench.core.pools.nested_subsets(n_pool, sizes, seed)` permutes the whole pool once with PCG64 and takes the
sorted first s indices, and seed 0 is his replicate 0, so our 10k, 30k and 100k are the same sequences as his draw. The files are written in
permutation order (not sorted), which makes every size a prefix of the next, the convention of the other pools in k562_nested, while the SETS are
identical to nested_subsets (asserted below). Only the 100k are ever scored.

    python scripts/program_oct22/label_nested_pool.py --src <pool.npz> --arm gpa_d3_student_mean [--seed 0]
    python scripts/program_oct22/label_nested_pool.py --src <pool.npz> --arm evo2 --reuse outputs/grid_pools/k562_cumulative/evo2_d100000.npz
      (--reuse takes the labels of an earlier labelling of the same sequences instead of calling the oracle; it fails on any sequence it does not know)

Collaborator pools stamp provenance in two shapes: one JSON `provenance` blob (diffusion pools) or separate `prov_*` scalars (Evo2 pool); both are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ALBENCH_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "experiments"))
from albench.core.pools import nested_subsets, verify_nesting  # noqa: E402

SIZES = (10_000, 30_000, 100_000)
NEST = R / "outputs/grid_pools/k562_nested"


def read_provenance(z) -> dict:
    if "provenance" in z.files:
        return json.loads(str(z["provenance"]))
    out = {}
    for k in z.files:
        if k.startswith("prov_"):
            v = z[k]
            try:
                v = v.item()
            except (ValueError, AttributeError):
                v = v.tolist() if hasattr(v, "tolist") else str(v)
            out[k[len("prov_") :]] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--arm", required=True, help="file stem of the nested pools")
    ap.add_argument(
        "--seed", type=int, default=0, help="sampling replicate of the collaborator's subsetting"
    )
    ap.add_argument(
        "--reuse",
        default="",
        help="npz with sequences + oracle_labels of the same sequences (skips the oracle)",
    )
    ap.add_argument("--out-dir", default=str(NEST))
    a = ap.parse_args()

    z = np.load(a.src, allow_pickle=True)
    seqs = np.array([str(s) for s in z["sequences"]], dtype=object)
    prov = read_provenance(z)
    n = len(seqs)
    print(f"source {a.arm}: {n:,} sequences", flush=True)

    subs = nested_subsets(n, SIZES, seed=a.seed)
    verify_nesting(subs)
    order = np.random.default_rng(a.seed).permutation(n)[: SIZES[-1]]
    for s in SIZES:
        assert set(order[:s].tolist()) == set(subs[s].tolist()), (
            f"permutation prefix {s} differs from nested_subsets"
        )
    top = seqs[order]
    assert len(set(top.tolist())) == len(top), (
        "duplicate sequences among the first 100k of the permutation"
    )
    extra = {k: np.asarray(z[k])[order] for k in ("student_scores", "member_std") if k in z.files}

    if a.reuse:
        old = np.load(a.reuse, allow_pickle=True)
        assert str(old["oracle_id"]) == "oracle_v3", old["oracle_id"]
        known = dict(
            zip((str(s) for s in old["sequences"]), np.asarray(old["oracle_labels"], np.float32))
        )
        labels = np.array([known[s] for s in top], dtype=np.float32)
        print(f"labels reused from {a.reuse} ({len(known):,} known sequences)", flush=True)
    else:
        os.environ.setdefault("AG_S2_ORACLE_DIR", str(R / "outputs/oracle_v3/s2"))
        from experiments.exp1_1_scaling import _load_oracle
        from scripts.generate_labeled_pools import _label_sequences

        oracle = _load_oracle("k562", oracle_type="ag_s2")
        labels = np.asarray(_label_sequences(oracle, list(top)), dtype=np.float32).ravel()
    assert len(labels) == len(top) and np.isfinite(labels).all(), "bad labels"
    print(f"labelled {len(labels):,}: mean {labels.mean():.3f}, sd {labels.std():.3f}", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = json.dumps(
        {
            "source": a.src,
            "subsetting": "albench.core.pools.nested_subsets",
            "replicate_seed": a.seed,
            "upstream_provenance": prov,
        }
    )
    for D in SIZES:
        path = out_dir / f"{a.arm}_d{D}.npz"
        tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
        np.savez(
            tmp,
            sequences=top[:D],
            oracle_labels=labels[:D],
            oracle_id=np.array("oracle_v3"),
            strategy=np.array(a.arm),
            seed=np.int64(a.seed),
            params=np.array(params),
            **{f"generator_{k}": v[:D] for k, v in extra.items()},
        )
        os.replace(tmp, path)
        print(f"wrote {path.name}: n={D}, label mean {labels[:D].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
