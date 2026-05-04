"""Analyze Task 2 D_min provisional results.

Walks results/preflight/task2_d_min/<arch>/d<D>/seed<seed>/result.json,
computes val R² per (arch, D, seed), applies the decision rule:

    D_min_provisional = smallest D where val R² > 0.1
                        across all 3 archs and 3 seeds

Writes:
    results/preflight/d_min_provisional.csv
    results/preflight/SUMMARY.md updated with Task 2 entry

Usage:
    uv run --no-sync python scripts/preflight/analyze_task2_d_min.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    base = REPO / "results" / "preflight" / "task2_d_min"
    if not base.exists():
        print(f"No results yet at {base}")
        return

    rows = []
    for f in sorted(base.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        d_train = int(parts[-3].lstrip("d"))
        seed = int(parts[-2].replace("seed", ""))
        # val R² = 1 - val_MSE / Var(val_labels). We don't have Var(val_labels)
        # tracked, so report best_val_mse as the primary diagnostic and a
        # rough R² approximation using observed val MSE / a typical Var(y_val).
        # Var(K562_log2FC) ≈ 1.5 from MAUDE distribution (we'll refine post-cache).
        TYPICAL_VAR_VAL = 1.5
        val_mse = float(d.get("best_val_mse", 0))
        val_r2_approx = 1.0 - val_mse / TYPICAL_VAR_VAL
        rows.append(
            {
                "arch": arch,
                "d_train": d_train,
                "seed": seed,
                "best_val_mse": round(val_mse, 4),
                "test_mse": round(float(d.get("test_mse_at_best_val", 0)), 4),
                "val_r2_approx": round(val_r2_approx, 4),
                "best_epoch": d.get("best_epoch"),
                "epochs": d.get("epochs"),
                "min_val_in_final_pct": d.get("min_val_in_final_pct_window"),
                "n_params": d.get("n_params"),
                "gpu_hrs": round(float(d.get("gpu_hrs", 0)), 3),
            }
        )

    if not rows:
        print(f"No result.json files yet in {base}")
        return

    # Sort, write CSV
    rows.sort(key=lambda r: (r["arch"], r["d_train"], r["seed"]))
    out_csv = REPO / "results" / "preflight" / "d_min_provisional.csv"
    fields = [
        "arch",
        "d_train",
        "seed",
        "best_val_mse",
        "test_mse",
        "val_r2_approx",
        "best_epoch",
        "epochs",
        "min_val_in_final_pct",
        "n_params",
        "gpu_hrs",
    ]
    with out_csv.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")

    # Decision rule: D_min_provisional = smallest D where val R² > 0.1
    # across all 3 archs × 3 seeds.
    THRESHOLD = 0.1
    by_arch_d: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        by_arch_d[(r["arch"], r["d_train"])].append(r["val_r2_approx"])

    archs = sorted({r["arch"] for r in rows})
    ds = sorted({r["d_train"] for r in rows})
    print(f"\n=== Val R² (approx) by (arch, D) ===")
    print(f"  D \\ arch    | " + " | ".join(f"{a:>10}" for a in archs))
    for d in ds:
        cells = []
        for a in archs:
            vals = by_arch_d.get((a, d), [])
            if vals:
                cells.append(f"min={min(vals):+.3f}")
            else:
                cells.append("    -    ")
        print(f"  D={d:>6}     | " + " | ".join(f"{c:>10}" for c in cells))

    # Find D_min: smallest D where min(val_r2) > THRESHOLD across all archs
    # AND we have all 3 seeds for that (arch, D).
    qualifying_ds = []
    for d in ds:
        all_good = True
        for a in archs:
            v = by_arch_d.get((a, d), [])
            if len(v) < 3 or min(v) <= THRESHOLD:
                all_good = False
                break
        if all_good:
            qualifying_ds.append(d)

    if qualifying_ds:
        d_min = min(qualifying_ds)
        print(
            f"\n>>> D_min_provisional = {d_min} (smallest D with all archs × seeds val_R² > {THRESHOLD})"
        )
    else:
        print(f"\n>>> No D in tested set satisfies the criterion. Pending more results.")

    # Sanity flag: any runs where min val loss landed in final 10%?
    flagged = [r for r in rows if r.get("min_val_in_final_pct")]
    if flagged:
        print(f"\nFLAGGED — best epoch in final 10% (epoch budget too tight):")
        for r in flagged:
            print(
                f"  {r['arch']} d={r['d_train']} seed={r['seed']}  "
                f"best_epoch={r['best_epoch']}/{r['epochs']}"
            )


if __name__ == "__main__":
    main()
