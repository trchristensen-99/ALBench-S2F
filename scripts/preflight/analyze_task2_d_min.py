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

    # Compute the true variance of val and test labels from the
    # ag_oracle pseudolabel cache (so val_R² / test_R² are real, not
    # estimated from a hardcoded TYPICAL_VAR_VAL).
    cache = REPO / "outputs" / "oracle_pseudolabels_k562_ag_s2_refalt"
    var_val = var_test = None
    if (cache / "val_oracle_labels.npz").exists():
        import numpy as _np

        var_val = float(_np.load(cache / "val_oracle_labels.npz")["true_label"].var())
        var_test = float(_np.load(cache / "test_oracle_labels.npz")["true_label"].var())
        print(f"  True Var(val) = {var_val:.4f}, Var(test) = {var_test:.4f}")
    else:
        print(f"  WARN: pseudolabel cache not at {cache}; falling back to TYPICAL_VAR=1.5")

    rows = []
    for f in sorted(base.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        d_train = int(parts[-3].lstrip("d"))
        seed = int(parts[-2].replace("seed", ""))
        val_mse = float(d.get("best_val_mse", 0))
        test_mse = float(d.get("test_mse_at_best_val", 0))
        # Real R² using the actual variance. R² ≈ 0 means "predicts the
        # mean" (no learning); negative means "worse than predicting mean".
        v_var = var_val if var_val is not None else 1.5
        t_var = var_test if var_test is not None else 1.5
        val_r2 = 1.0 - val_mse / v_var
        test_r2 = 1.0 - test_mse / t_var
        rows.append(
            {
                "arch": arch,
                "d_train": d_train,
                "seed": seed,
                "best_val_mse": round(val_mse, 4),
                "test_mse": round(test_mse, 4),
                "val_r2": round(val_r2, 4),
                "test_r2": round(test_r2, 4),
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
        "val_r2",
        "test_r2",
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

    # Decision rule: D_min_provisional = smallest D where TEST R² > 0.1
    # (using TRUE Var(test_labels)) across all 3 archs × 3 seeds. Test R²
    # is the right metric for scaling laws because the main sweep fits
    # MSE_k(D) on the test set; val R² is logged for cross-check only.
    THRESHOLD = 0.1
    by_arch_d_test: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_arch_d_val: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        by_arch_d_test[(r["arch"], r["d_train"])].append(r["test_r2"])
        by_arch_d_val[(r["arch"], r["d_train"])].append(r["val_r2"])

    archs = sorted({r["arch"] for r in rows})
    ds = sorted({r["d_train"] for r in rows})

    def _print_table(title: str, table: dict[tuple[str, int], list[float]]):
        print(f"\n=== {title} ===")
        print(f"  D \\ arch    | " + " | ".join(f"{a:>14}" for a in archs))
        for d in ds:
            cells = []
            for a in archs:
                vals = table.get((a, d), [])
                if vals:
                    cells.append(f"min={min(vals):+.3f}")
                else:
                    cells.append("    -    ")
            print(f"  D={d:>6}     | " + " | ".join(f"{c:>14}" for c in cells))

    _print_table("Test R² (TRUE variance) — primary metric", by_arch_d_test)
    _print_table("Val R² (TRUE variance) — secondary diagnostic", by_arch_d_val)

    # Find D_min: smallest D where min(test_r2) > THRESHOLD across all archs
    # AND we have all 3 seeds for that (arch, D).
    qualifying_ds = []
    for d in ds:
        all_good = True
        for a in archs:
            v = by_arch_d_test.get((a, d), [])
            if len(v) < 3 or min(v) <= THRESHOLD:
                all_good = False
                break
        if all_good:
            qualifying_ds.append(d)

    if qualifying_ds:
        d_min = min(qualifying_ds)
        print(
            f"\n>>> D_min_provisional = {d_min} "
            f"(smallest D with all archs × seeds test_R² > {THRESHOLD})"
        )
    else:
        print(
            f"\n>>> No D in tested set satisfies test_R² > {THRESHOLD} — "
            f"D_min must be higher than {max(ds)}. The scaling-law fit cannot use "
            f"these points. Recommend extending d_grid upward and rerunning Task 2."
        )

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
