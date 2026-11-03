#!/usr/bin/env python
"""Results of the Nov-2 schedule-length / early-stopping arms (`lr39_`, native LegNet trainer) and of the harness-parity arms (`pa38_ref_`, the published LegNet through the loop/menu harness).

For every directory: number of models, mean single-model test r on genomic / OOD / random (oracle labels), mean validation r (V0c) and the epochs actually trained (best epoch for the native trainer, epochs trained for the harness),
next to the 20-seed banks of the same recipe family. Single-model means, never stacks: the arms have 3 to 5 seeds.

    python scripts/analysis/analyze_lr39_pa38.py      (cpuq, venv python)
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

MR = "outputs/menu_retrain_v0"
SETS = ("genomic", "ood", "random_32k")


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def summarise(d: str) -> dict | None:
    p = f"{MR}/{d}"
    if not os.path.exists(f"{p}/labels.npz"):
        return None
    lab = np.load(f"{p}/labels.npz")
    rs = {s: [] for s in SETS}
    val, ep = [], []
    for mf in sorted(glob.glob(f"{p}/r*_meta.json")):
        try:
            m = json.load(open(mf))
            z = np.load(mf.replace("_meta.json", ".npz"))
            for s in SETS:
                rs[s].append(pearson(z[f"test_pred_{s}"], lab[f"oracle_{s}"]))
            val.append(pearson(z["val_pred"], lab["val_labels"]))
            ep.append(float(m.get("best_epoch") or m.get("epochs_trained") or np.nan))
        except Exception:
            continue
    if not val:
        return None
    return {
        "n": len(val),
        **{s: float(np.mean(v)) for s, v in rs.items()},
        "val": float(np.mean(val)),
        "epochs": float(np.nanmean(ep)),
    }


def row(label: str, r: dict | None) -> None:
    if r is None:
        print(f"  {label:44s} (no models)")
        return
    print(
        f"  {label:44s} n={r['n']:2d}  genomic {r['genomic']:.4f}  OOD {r['ood']:.4f}  random {r['random_32k']:.4f}  val {r['val']:.4f}  epochs {r['epochs']:.0f}"
    )


def main() -> None:
    print("Native trainer, schedule length and early stopping (single-model means)")
    groups = {
        "10k": [
            "bench38_ln_e100_10k_c20",
            "bench38_ln_e300_wd2_10k_c20",
            "bench38_ln_e500_wd4_10k_c20",
            "bench38_ln_e200_wd4_10k_c20",
            "bench38_ln_e300_wd4_10k_c20",
            "bench38_ln_e800_wd4_10k_c20",
            "bench38_ln_e800_wd4_es16_10k_c20",
        ],
        "30k": [
            "bench38_ln_e100_30k_c20",
            "bench38_ln_e150_wd2_30k_c20",
            "bench38_ln_e200_wd2_30k_c20",
            "bench38_ln_e300_wd2_30k_c20",
            "bench38_ln_e500_wd2_30k_c20",
            "bench38_ln_e500_wd2_es16_30k_c20",
        ],
        "100k": [
            "bench38_ln_e100_100k_c20",
            "bench38_ln_e200_100k_c20",
            "bench38_ln_e300_100k_c20",
            "bench38_ln_e500_100k_c20",
            "bench38_ln_e500_es16_100k_c20",
        ],
    }
    for size, dirs in groups.items():
        print(f"{size}:")
        for d in dirs:
            row(d.replace("bench38_ln_", "").replace(f"_{size}_c20", ""), summarise(d))
    print(
        "\nHarness parity: the published LegNet through the menu-library harness (patience off, fp16) against the native bank"
    )
    for D, tuned in ((10, "e800_wd4"), (30, "e500_wd2"), (100, "e500")):
        print(f"{D}k:")
        row("native e100 bank", summarise(f"bench38_ln_e100_{D}k_c20"))
        row(f"native tuned bank ({tuned})", summarise(f"bench38_ln_{tuned}_{D}k_c20"))
        row("harness, both recipes pooled (pa38_ref)", summarise(f"pa38_ref_{D}k"))
        # per recipe: entries are named legnet_ref_e100_<D>k and legnet_ref_tuned_<D>k
        p = f"{MR}/pa38_ref_{D}k"
        if os.path.exists(f"{p}/labels.npz"):
            lab = np.load(f"{p}/labels.npz")
            by: dict = {}
            for mf in sorted(glob.glob(f"{p}/r*_meta.json")):
                try:
                    m = json.load(open(mf))
                    z = np.load(mf.replace("_meta.json", ".npz"))
                    by.setdefault(m["menu_entry"], []).append(
                        (
                            pearson(z["test_pred_genomic"], lab["oracle_genomic"]),
                            pearson(z["test_pred_ood"], lab["oracle_ood"]),
                            pearson(z["val_pred"], lab["val_labels"]),
                            float(m.get("epochs_trained") or np.nan),
                        )
                    )
                except Exception:
                    continue
            for k, v in sorted(by.items()):
                a = np.array(v)
                print(
                    f"  {'harness ' + k:44s} n={len(v):2d}  genomic {a[:, 0].mean():.4f}  OOD {a[:, 1].mean():.4f}  val {a[:, 2].mean():.4f}  epochs {np.nanmean(a[:, 3]):.0f}"
                )


if __name__ == "__main__":
    main()
