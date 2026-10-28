"""Turn a freshly built 100k pool (scripts/build_pool.py) into nested canonical pools at 10k/30k/100k, and the AL-experiment top-up pool.

Why a permutation: build_pool.py writes sequences in generation order, and some strategies generate in batches (random_gc_sweep makes one batch per GC
level), so a plain prefix of the raw pool would be a single batch. The seeded permutation makes every prefix a fair draw, and the files are nested by
construction (10k within 30k within 100k), like the other pools in k562_nested.

    python scripts/program_oct22/slice_nested_new.py --strategy random_gc_sweep --name random_gc_sweep --raw outputs/grid_pools/k562_reps_raw100k
    python scripts/program_oct22/slice_nested_new.py --strategy mutagenesis --name mutagenesis_mr10 --set mutation_rate=0.1 --raw ...

Also writes k562_acq/res_<name>_d100000.npz = the 30k genomic base (replicate 1) + the first 70k of the nested 100k pool, the training set of the
"30k genomic + 70k from one reservoir" arm of the acquisition experiment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ALBENCH_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(R))
from albench.pools import PoolSpec  # noqa: E402

GP = R / "outputs/grid_pools"
SIZES = (10_000, 30_000, 100_000)


def save(path: Path, seqs, labels, strategy: str, seed: int) -> None:
    tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(
        tmp,
        sequences=np.asarray(seqs, dtype=object),
        oracle_labels=np.asarray(labels, np.float32),
        oracle_id="oracle_v3",
        strategy=strategy,
        seed=np.int64(seed),
    )
    os.replace(tmp, path)


def gc(seqs) -> np.ndarray:
    return np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--name", required=True, help="file stem of the nested pool")
    ap.add_argument("--raw", required=True, help="out dir that build_pool.py wrote to")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    from scripts.build_pool import _coerce

    params = {k.strip(): _coerce(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    spec = PoolSpec(
        strategy=args.strategy,
        params=params,
        size=100_000,
        generation_seed=args.seed,
        task="k562",
    )
    z = np.load(spec.path(args.raw), allow_pickle=True)
    seqs, lab = z["sequences"], z["oracle_labels"].astype(np.float32)
    assert str(z["oracle_id"]) == "oracle_v3" and len(seqs) >= 100_000, (z["oracle_id"], len(seqs))
    assert np.isfinite(lab).all()
    perm = np.random.default_rng(args.seed).permutation(len(seqs))[:100_000]
    seqs, lab = seqs[perm], lab[perm]
    assert len(set(seqs.tolist())) == len(seqs), "duplicate sequences in the pool"
    nested = GP / "k562_nested"
    for D in SIZES:
        save(nested / f"{args.name}_d{D}.npz", seqs[:D], lab[:D], args.name, args.seed)
        g = gc(seqs[:D])
        print(
            f"{args.name} d{D}: label mean {lab[:D].mean():.3f}, GC mean {g.mean():.3f} (levels seen: {np.unique(np.round(g, 2)).size})",
            flush=True,
        )
    base = np.load(GP / "k562_reps/genomic_r1_d30000.npz", allow_pickle=True)
    acq = GP / "k562_acq"
    acq.mkdir(exist_ok=True)
    save(
        acq / f"res_{args.name}_d100000.npz",
        list(base["sequences"][:30_000]) + list(seqs[:70_000]),
        np.concatenate([base["oracle_labels"][:30_000].astype(np.float32), lab[:70_000]]),
        f"res_{args.name}",
        args.seed,
    )
    print("wrote", acq / f"res_{args.name}_d100000.npz")


if __name__ == "__main__":
    main()
