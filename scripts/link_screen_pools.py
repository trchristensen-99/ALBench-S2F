"""Expose the flat screen cache in the {reservoir}/pool.npz layout exp1_1_scaling wants.

The screen writes one file per cell, flat and self-describing:
    outputs/screen/cache/<strategy>__d30000__seed<k>__<param>-<val>__labeled.npz
The scaling driver resolves pools as {pool_base_dir}/{reservoir}/pool.npz.

Symlinks, not copies: 105 cells x 30k sequences is ~600MB that would otherwise be
duplicated for nothing, and a copy silently goes stale if a cell is relabelled.
Each cell is its own "reservoir" here -- the cell IS the (strategy, params, seed)
identity the screen is measuring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="outputs/screen/cache")
    ap.add_argument("--out", default="outputs/screen/pools")
    ap.add_argument("--d", type=int, default=30000, help="only link cells at this size")
    ap.add_argument(
        "--glob",
        default=None,
        help="explicit glob for labelled files, overriding --d. The link name is the "
        "filename with __labeled and any __n<size>__seed<k> suffix stripped, which is "
        "what the driver resolves as the reservoir name.",
    )
    args = ap.parse_args()

    cache = REPO / args.cache
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    cells = sorted(cache.glob(args.glob or f"*__d{args.d}__*__labeled.npz"))
    if not cells:
        print(f"no labelled cells at d={args.d} under {cache}", file=sys.stderr)
        return 1

    manifest, skipped = {}, []
    for c in cells:
        name = c.name.replace("__labeled.npz", "")
        if args.glob:
            # Curve pools are <reservoir>__n<size>__seed<k>; the driver looks them up
            # by reservoir name alone, so strip the size/seed suffix.
            import re as _re
            name = _re.sub(r"__n\d+__seed\d+$", "", name)
        z = np.load(c, allow_pickle=True)
        if "oracle_labels" not in z.files and "labels" not in z.files:
            skipped.append((name, "no label array"))
            continue
        y = z["oracle_labels"] if "oracle_labels" in z.files else z["labels"]
        if not np.isfinite(y).all():
            skipped.append((name, "non-finite labels"))
            continue
        d = out / name
        d.mkdir(exist_ok=True)
        link = d / "pool.npz"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(c.resolve())
        manifest[name] = {
            "cell": str(c.relative_to(REPO)),
            "n": int(len(y)),
            "oracle_id": str(z["oracle_id"]) if "oracle_id" in z.files else None,
            "strategy": str(z["strategy"]) if "strategy" in z.files else None,
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"linked {len(manifest)} cells into {out}")
    if skipped:
        print(f"SKIPPED {len(skipped)}:")
        for n, why in skipped:
            print(f"  {n}: {why}")
    oids = {v["oracle_id"] for v in manifest.values()}
    print(f"oracle_id(s) across cells: {oids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
