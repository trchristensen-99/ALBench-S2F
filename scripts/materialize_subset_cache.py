"""Materialize a seeded D-subset of a master pool into a reservoir-cache npz.

The HP search consumes pools via `--reservoir_cache <path>` expecting keys
`sequences` + `oracle_labels`. This draws a size-D *seeded random* subset from a
strategy's master pool (labeled once with full856k_clean) and writes that exact
schema, so nothing downstream changes — only the provenance improves and the
subset is no longer a biased fixed prefix.

Usage::
    uv run --no-sync python scripts/materialize_subset_cache.py \
        --task k562 --reservoir prm_5pct --D 30000 --seed 42 \
        --out outputs/reservoir_cache/k562_prm_5pct_d30000_seed42.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.master_pool_io import draw_subset, load_master  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="k562", choices=["k562", "yeast"])
    ap.add_argument("--reservoir", required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--master-root", type=Path, default=None)
    args = ap.parse_args()

    seqs, labels = load_master(args.task, args.reservoir, args.master_root)
    if args.D > len(seqs):
        print(
            f"WARNING: requested D={args.D:,} > master pool {len(seqs):,} for "
            f"{args.reservoir}; returning the whole pool",
            flush=True,
        )
    sub_seqs, sub_labels = draw_subset(seqs, labels, args.D, args.seed)

    meta = {
        "task": args.task,
        "reservoir": args.reservoir,
        "D": int(args.D),
        "seed": int(args.seed),
        "n": int(len(sub_seqs)),
        "master_n": int(len(seqs)),
        "label_mean": float(np.mean(sub_labels)),
        "label_std": float(np.std(sub_labels)),
        "source": "master_pool_seeded_subset",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        sequences=np.array(sub_seqs, dtype=object),
        oracle_labels=sub_labels.astype(np.float32),
        oracle_id=np.array("full856k_clean"),
        metadata=json.dumps(meta),
    )
    print(
        f"Wrote {args.out}  n={len(sub_seqs):,}/{len(seqs):,}  "
        f"mean={meta['label_mean']:.3f} std={meta['label_std']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
