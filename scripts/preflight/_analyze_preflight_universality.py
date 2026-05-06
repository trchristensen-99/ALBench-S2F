"""Analyze pre-flight task results for universality + edge-of-grid concerns.

Three questions:
  Q1. Are augmentations universally better, or only for some N / some archs?
      (task5_augmentations is no_aug, rev_complement, rc_shift, rc_shift_evoaug)
  Q2. Do we have HP grid edges (LR or BS at the boundary of what we tested)?
      Should we expand the task3 grid?
  Q3. Was task3's HP search done WITHOUT augmentations? If task5 winners
      change the optimal HPs, we may need to re-run task3 with the
      winning aug.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def load_task_results(task_dir: Path) -> pd.DataFrame:
    rows = []
    for rj in sorted(task_dir.rglob("result.json")):
        try:
            d = json.loads(rj.read_text())
            d["_path"] = str(rj.relative_to(REPO))
            rows.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {rj}: {e}")
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("Q1. Augmentations universality (Task 5)")
    print("=" * 80)
    df5 = load_task_results(REPO / "results/preflight/task5_augmentations")
    if df5.empty:
        print("  no task5 results")
    else:
        # Group by arch × augmentation, average over seeds
        agg = (
            df5.groupby(["arch", "augmentations"])["test_mse_at_best_val"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg.columns = ["arch", "aug", "test_mse_mean", "test_mse_std", "n_seeds"]
        # Pivot so we can compare side-by-side
        pivot = agg.pivot(index="arch", columns="aug", values="test_mse_mean")
        print("\n  Mean test_mse per (arch, augmentation):")
        print(pivot.to_string(float_format="%.4f"))
        print("\n  Best augmentation per arch (lowest test_mse):")
        for arch in pivot.index:
            best = pivot.loc[arch].idxmin()
            print(f"    {arch}: {best} ({pivot.loc[arch, best]:.4f})")
        # Universality check: does the best aug agree across archs?
        bests = {arch: pivot.loc[arch].idxmin() for arch in pivot.index}
        if len(set(bests.values())) == 1:
            print(f"  ✓ Universal best aug: {next(iter(bests.values()))}")
        else:
            print(f"  ⚠ Best aug DIFFERS by arch: {bests}")

    print("\n" + "=" * 80)
    print("Q2. HP grid edges (Task 3)")
    print("=" * 80)
    df3 = load_task_results(REPO / "results/preflight/task3_lr_bs")
    if df3.empty:
        print("  no task3 results")
    else:
        # For each arch, find which (lr, bs) cell won
        df3["lr"] = df3["hp"].apply(lambda h: h.get("lr"))
        df3["bs"] = df3["hp"].apply(lambda h: h.get("batch_size"))
        for arch in df3["arch"].unique():
            sub = df3[df3["arch"] == arch]
            grid = (
                sub.groupby(["lr", "bs"])["test_mse_at_best_val"]
                .mean()
                .reset_index()
                .sort_values("test_mse_at_best_val")
            )
            best = grid.iloc[0]
            lr_unique = sorted(sub["lr"].unique())
            bs_unique = sorted(sub["bs"].unique())
            lr_pos = lr_unique.index(best["lr"])
            bs_pos = bs_unique.index(best["bs"])
            at_lr_edge = lr_pos == 0 or lr_pos == len(lr_unique) - 1
            at_bs_edge = bs_pos == 0 or bs_pos == len(bs_unique) - 1
            edge_flag = "⚠ AT EDGE" if (at_lr_edge or at_bs_edge) else "✓ INTERIOR"
            print(
                f"  {arch}: best lr={best['lr']:.0e} (pos {lr_pos + 1}/{len(lr_unique)} in {lr_unique})  "
                f"bs={int(best['bs'])} (pos {bs_pos + 1}/{len(bs_unique)} in {bs_unique})  "
                f"→ {edge_flag}"
            )
            # Show top-5 cells per arch
            print(f"    Top-5 cells:")
            for _, r in grid.head(5).iterrows():
                print(
                    f"      lr={r['lr']:.0e} bs={int(r['bs'])} → test_mse={r['test_mse_at_best_val']:.4f}"
                )

    print("\n" + "=" * 80)
    print("Q3. Was task3 done WITHOUT augmentations?")
    print("=" * 80)
    if not df3.empty:
        augs = df3["augmentations"].value_counts().to_dict()
        print(f"  task3 augmentation distribution: {augs}")
        if len(augs) == 1 and "none" in augs:
            print("  ⚠ task3 used NO augmentations — if task5 winner is shift/RC, may need rerun")
        elif len(augs) == 1:
            print(f"  task3 used a single augmentation: {list(augs.keys())[0]}")
        else:
            print(f"  task3 mixed augmentations: {augs}")

    print("\n" + "=" * 80)
    print("Q4. Does the optimal aug differ by dataset size?")
    print("=" * 80)
    # task5 was at d=600k only — but task6 d=500/600k may show aug interactions
    df6 = load_task_results(REPO / "results/preflight/task6_parameterization")
    if not df6.empty:
        df6["d_train"] = df6["d_train"].astype(int)
        d_unique = sorted(df6["d_train"].unique())
        print(f"  task6 d_train values: {d_unique}")
        if "augmentations" in df6.columns:
            for d in d_unique:
                sub = df6[df6["d_train"] == d]
                augs = sub["augmentations"].value_counts().to_dict()
                print(f"    d={d}: aug usage = {augs}")
    if not df5.empty:
        ds = df5["d_train"].unique() if "d_train" in df5.columns else []
        print(f"\n  task5 d_train values: {sorted(ds)}")


if __name__ == "__main__":
    main()
