"""Build a menu.json of candidate architectures from AutoResearch cells.

Selection mirrors the intended pipeline: the top-k models by VALIDATION Pearson from each independent AutoResearch instance
(cell) are the candidate pool. The reference LegNet-eff network (the one shown to the proposer in phase 1) is added as a fixed baseline.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True, help="AutoResearch output dirs (each = one independent instance)")
    ap.add_argument("--k", type=int, default=8, help="top-k by val per instance")
    ap.add_argument("--min_val", type=float, default=0.0)
    ap.add_argument("--first_n", type=int, default=0, help="only the first N productive models of each run are candidates (0 = all)")
    ap.add_argument("--no_reference", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    menu, seen = [], set()
    for cell in args.cells:
        rows = []
        for f in glob.glob(cell.rstrip("/") + "/r*_arch_*_meta.json"):
            try:
                m = json.load(open(f))
            except Exception:
                continue
            if args.first_n and int(m["model_id"].split("_")[-1]) >= args.first_n:
                continue
            if m.get("source") and m.get("train_config") and m.get("val_pearson") is not None and m["val_pearson"] >= args.min_val:
                rows.append(m)
        rows.sort(key=lambda m: -m["val_pearson"])
        name = "/".join(Path(cell.rstrip("/")).parts[-2:])
        for m in rows[:args.k]:
            key = m["source"].strip()
            if key in seen:  # identical code from two instances: keep one
                continue
            seen.add(key)
            menu.append({"id": f"{name.replace('/', '_')}_{m['model_id']}", "cell": name, "idea": m.get("idea", ""),
                         "orig_val": m["val_pearson"], "n_params": m.get("n_params"), "source": m["source"],
                         "train_config": m["train_config"]})
        print(f"  {name}: {len(rows)} usable models, took top {min(args.k, len(rows))}")
    if not args.no_reference:
        from albench.arch_search.arch_autoresearch import REFERENCE_SRC
        menu.append({"id": "reference_legnet_eff", "cell": "reference", "idea": "LegNet-style EffBlock reference (shown to proposer in phase 1)",
                     "orig_val": None, "n_params": None, "source": REFERENCE_SRC,
                     "train_config": {"optimizer": "muon", "lr": 0.002, "weight_decay": 0.0001, "batch_size": 256,
                                      "epochs": 100, "lr_schedule": "onecycle"}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(menu, open(args.out, "w"), indent=1)
    print(f"menu: {len(menu)} entries -> {args.out}")


if __name__ == "__main__":
    main()
