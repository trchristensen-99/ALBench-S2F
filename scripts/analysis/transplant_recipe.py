#!/usr/bin/env python
"""Give every architecture of a menu the training recipe of the published MPRA-LegNet, to separate the two things that differ between a menu member and the tuned LegNet at large D.

At 100k the tuned LegNet (AdamW, peak lr 0.01, weight decay 0.1, batch 1024, 100 epochs, one-cycle) reaches 0.9307 genomic as a 10-seed ensemble against 0.9205 for the LLM menu v2, and a single seed
0.9206 against 0.884: the menu members were found at 30k with Muon, lr 2e-3 and batch 256. If the architectures of the menu keep their accuracy under the LegNet recipe the gap is the recipe; if they lose it
the gap is the architecture.

    python scripts/analysis/transplant_recipe.py --menu outputs/menus_oct30/V2.json --out outputs/menus_oct31/V2_e100.json [--tag e100]
"""

from __future__ import annotations

import argparse
import copy
import json
import os

RECIPES = {
    "e100": {
        "optimizer": "adamw",
        "lr": 0.01,
        "weight_decay": 0.1,
        "batch_size": 1024,
        "epochs": 100,
        "lr_schedule": "onecycle",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="e100", choices=sorted(RECIPES))
    ap.add_argument(
        "--one_per_arch",
        action="store_true",
        help="menus that hold the same architecture under several recipes (PI.json) keep only the first entry of each architecture",
    )
    a = ap.parse_args()
    entries = json.load(open(a.menu))
    out, seen = [], set()
    for e in entries:
        key = e["source"]
        if a.one_per_arch and key in seen:
            continue
        seen.add(key)
        e = copy.deepcopy(e)
        e["train_config"] = dict(RECIPES[a.tag])
        e["id"] = f"{e['id']}_{a.tag}"
        e["note"] = f"training recipe replaced by the {a.tag} recipe of the published MPRA-LegNet"
        out.append(e)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(out), "entries from", len(entries))


if __name__ == "__main__":
    main()
