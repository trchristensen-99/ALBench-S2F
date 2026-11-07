"""Extend the mutagenesis and motif_shared_core replicate pools to 300k, keeping every size a prefix.

The 300k rung needed three reservoirs that nobody had generated at that size. `scripts/build_pool.py` produced them
(`k562_reps_raw300k_new/`), but a fresh 300k draw is not automatically a superset of the 10k/30k/100k pools those
reservoirs already ran on: for `motif_shared_core` the generator happens to be prefix-stable and it is, while for
`mutagenesis` the 300k draw shares nothing with the existing 100k pool. Writing it as `_d300000.npz` would make the
300k point of the curve a different sample rather than more of the same one, which is the one thing the nested
design exists to prevent.

So mutagenesis is extended instead of replaced: the existing 100k pool stays as the first 100k rows and the new draw
supplies the remaining 200k, skipping any sequence already present. Both halves come from the same generator at the
same generation seed and carry oracle_v3 labels, so the result is still a draw from that reservoir -- it is only the
ORDER that is pinned, which is what nesting needs.

`zoonomia_orthologs` is deliberately not written here. Its draws are 15.4% exact duplicates at every size (identical
orthologous windows across species), so a 300k draw holds ~253k distinct sequences and even unioned with the existing
30k pool it cannot reach 300k distinct. That is a property of the reservoir, not of this script, and it is reported
rather than papered over.

    python scripts/extend_replicate_pools_300k.py [--dry]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GP = REPO / "outputs/grid_pools"
RAW = GP / "k562_reps_raw300k_new"
OUT = GP / "k562_reps"
SEED0 = {1: 43, 2: 44, 3: 45}
TARGET = 300_000
# base size = the largest existing prefix that must survive as the head of the 300k pool
PLAN = {"motif_shared_core": 100_000, "mutagenesis": 100_000}


def load(path: Path) -> tuple[list[str], np.ndarray, str]:
    z = np.load(path, allow_pickle=True)
    return list(z["sequences"]), z["oracle_labels"].astype(np.float32), str(z["oracle_id"])


def save(path: Path, seqs, labels, oracle_id: str, strategy: str, seed: int) -> None:
    tmp = path.with_suffix(f".tmp{path.stem}.npz")
    np.savez(
        tmp,
        sequences=np.asarray(seqs, dtype=object),
        oracle_labels=np.asarray(labels, np.float32),
        oracle_id=oracle_id,
        strategy=strategy,
        seed=np.int64(seed),
    )
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    report = {}
    for strat, base_n in PLAN.items():
        for rep, gseed in SEED0.items():
            dest = OUT / f"{strat}_r{rep}_d{TARGET}.npz"
            raw = sorted(glob.glob(str(RAW / f"{strat}__g{gseed}" / "*.npz")))
            if not raw:
                print(f"  {strat} r{rep}: no raw 300k draw")
                continue
            if dest.exists():
                print(f"  {strat} r{rep}: {dest.name} already exists, left alone")
                continue
            base_path = OUT / f"{strat}_r{rep}_d{base_n}.npz"
            bs, bl, boid = load(base_path)
            ns, nl, noid = load(Path(raw[0]))
            assert boid == noid, (boid, noid)
            prefix_ok = len(ns) >= len(bs) and all(
                bs[i] == ns[i] for i in (0, 1, len(bs) // 2, len(bs) - 1)
            )
            if prefix_ok:
                seqs, labels, how = ns[:TARGET], nl[:TARGET], "sliced (draw is already a superset)"
            else:
                have = set(bs)
                keep = [i for i, s in enumerate(ns) if s not in have][: TARGET - len(bs)]
                seqs = bs + [ns[i] for i in keep]
                labels = np.concatenate([bl, nl[keep]])
                how = f"extended ({len(bs)} existing + {len(keep)} novel)"
            n_uniq = len(set(seqs))
            print(
                f"  {strat} r{rep}: {len(seqs)} rows, {n_uniq} distinct -- {how}"
                + ("  [DRY]" if args.dry else "")
            )
            report[f"{strat}_r{rep}"] = {"n": len(seqs), "n_unique": n_uniq, "how": how}
            if len(seqs) < TARGET:
                print("    short of 300k; not written")
                continue
            if not args.dry:
                save(dest, seqs, labels, boid, strat, gseed)
    if not args.dry and report:
        p = GP / "k562_reps/EXTEND_300K.json"
        p.write_text(json.dumps(report, indent=1))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
