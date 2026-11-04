#!/usr/bin/env python
"""Merge the shards written by `yash_menu_retrain.py`, fit the stack (`ensemble.json`, same format as the shared genomic_{10k,30k}_patience16) and score it on the three headline test sets.

    python scripts/yash_finish_ensemble.py --D 100000 --shards outputs/yash_ens/genomic_100k_patience16_shards --out outputs/yash_ens/genomic_100k_patience16

Writes `OUT/members/*`, `OUT/ensemble.json` and `OUT/RESULTS.json` (ensemble validation r and test r on genomic / ood / random_32k, member counts). Copying to the shared folder is a separate, deliberate step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--menu", default="v0")
    a = ap.parse_args()

    from albench.ensemble import menu as M
    from albench.paths import resolve

    out = Path(a.out)
    (out / "members").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(a.shards).glob("s*/members/*")):
        shutil.copy2(f, out / "members" / f.name)
        n += f.suffix == ".json"
    print(f"merged {n} members from {a.shards}", flush=True)
    menu = M.load_menu(a.menu)
    val_s, val_y = M.load_labelled(resolve("val_v0c"))
    meta = M.stack_dir(menu, out, val_y, size=a.D, screen=True, device="cuda:0")
    kept = sum(1 for m in meta["members"] if m.get("weight", 0) > 0)
    print(
        f"stacked {len(meta['members'])} members ({kept} with positive weight); validation r {meta['val_r_ensemble']:.4f}",
        flush=True,
    )
    ens = M.MenuEnsemble.load(out, val_s, val_y, menu=a.menu, devices=["cuda:0"])
    res = M.evaluate(ens)
    print("test r: " + ", ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
    (out / "RESULTS.json").write_text(
        json.dumps(
            {
                "size": a.D,
                "menu": a.menu,
                "members": len(meta["members"]),
                "positive_weight": kept,
                "val_r_ensemble": meta["val_r_ensemble"],
                "test_r": res,
                "screened_out": meta.get("screened_out"),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
