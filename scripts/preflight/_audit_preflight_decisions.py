"""Audit pre-flight decisions for:
- Q1. LegNet aug fairness — was aug=rev_complement HP grid actually
  tested at LegNet's best HPs, or just at task3's old HPs?
- Q2. Are any locked optima at grid edges?
- Q3. Are decisions tested across D values, or only at d=600k?
- Q4. Latest debias eval status.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    for rj in path.rglob("result.json"):
        try:
            d = json.loads(rj.read_text())
            d["_path"] = str(rj.relative_to(REPO))
            rows.append(d)
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("Q1. Is LegNet aug=none truly best, or did old HPs disadvantage aug?")
    print("=" * 80)
    df3 = load_results(REPO / "results/preflight/task3_lr_bs")
    df5 = load_results(REPO / "results/preflight/task5_augmentations")
    df3r_legnet = load_results(REPO / "results/preflight/task3_retry_legnet_noaug")
    print("\n  Task3 (orig grid, aug=rev_complement) LegNet best cells:")
    if not df3.empty:
        sub = df3[df3["arch"] == "legnet"].copy()
        sub["lr"] = sub["hp"].apply(lambda h: h.get("lr"))
        sub["bs"] = sub["hp"].apply(lambda h: h.get("batch_size"))
        sub = sub.sort_values("test_mse_at_best_val").head(5)
        for _, r in sub.iterrows():
            print(
                f"    lr={r['lr']:.0e} bs={int(r['bs'])} aug={r['augmentations']} test_mse={r['test_mse_at_best_val']:.4f}"
            )
    print("\n  Task5 (aug ablation, locked HPs from task3) LegNet:")
    if not df5.empty:
        sub5 = (
            df5[df5["arch"] == "legnet"]
            .groupby("augmentations")["test_mse_at_best_val"]
            .agg(["mean", "min", "max", "count"])
        )
        print(sub5.to_string(float_format="%.4f"))
    print("\n  Task3-retry (aug=none, extended LR) LegNet best 5:")
    if not df3r_legnet.empty:
        sub = df3r_legnet.copy()
        sub["lr"] = sub["hp"].apply(lambda h: h.get("lr"))
        sub["bs"] = sub["hp"].apply(lambda h: h.get("batch_size"))
        sub = sub.sort_values("test_mse_at_best_val").head(5)
        for _, r in sub.iterrows():
            print(
                f"    lr={r['lr']:.0e} bs={int(r['bs'])} aug={r['augmentations']} test_mse={r['test_mse_at_best_val']:.4f}"
            )
    print("\n  ⚠ MISSING: task3-retry with aug=rev_complement + extended LR for LegNet.")
    print("     If aug=rev_complement at lr=3e-3 (the no-aug winner) gave 0.16, our")
    print("     conclusion that 'aug hurts LegNet' would be wrong. Need a small confirm.")

    print("\n" + "=" * 80)
    print("Q2. Edges in task5/6/7 grids")
    print("=" * 80)
    df6 = load_results(REPO / "results/preflight/task6_parameterization")
    df7 = load_results(REPO / "results/preflight/task7_dropout")
    print("\n  Task6 (size: half/default/double) — best size per arch:")
    if not df6.empty:
        for arch in df6["arch"].unique():
            sub = df6[df6["arch"] == arch]
            agg = sub.groupby(sub["hp"].apply(lambda h: h.get("size_label", "?")))[
                "test_mse_at_best_val"
            ].mean()
            best_sz = agg.idxmin()
            sizes_tested = list(agg.index)
            edge = (
                "⚠ EDGE"
                if best_sz in (sizes_tested[0], sizes_tested[-1]) and len(sizes_tested) > 2
                else "✓"
            )
            print(f"    {arch}: tested {sizes_tested}, best={best_sz} ({agg[best_sz]:.4f}) {edge}")

    print("\n  Task7 (dropout) — best dropout per arch:")
    if not df7.empty:
        for arch in df7["arch"].unique():
            sub = df7[df7["arch"] == arch]
            sub = sub.copy()
            # Find dropout key by arch
            if arch == "legnet":
                sub["dropout"] = sub["hp"].apply(lambda h: h.get("dropout"))
            elif arch == "dream_rnn":
                sub["dropout"] = sub["hp"].apply(lambda h: h.get("dropout_lstm"))
            elif arch == "dream_attn":
                sub["dropout"] = sub["hp"].apply(lambda h: h.get("core_dropout"))
            agg = sub.groupby("dropout")["test_mse_at_best_val"].mean()
            tested = sorted(agg.index)
            best_dr = agg.idxmin()
            best_pos = tested.index(best_dr)
            edge = "⚠ EDGE" if best_pos == 0 or best_pos == len(tested) - 1 else "✓ INTERIOR"
            print(
                f"    {arch}: tested {tested}, best={best_dr} (mse={agg[best_dr]:.4f}, pos {best_pos + 1}/{len(tested)}) {edge}"
            )

    print("\n" + "=" * 80)
    print("Q3. Are decisions tested across D values? (universality across N)")
    print("=" * 80)
    for label, df_ in [("task3", df3), ("task5", df5), ("task6", df6), ("task7", df7)]:
        if df_.empty:
            continue
        ds = sorted(df_["d_train"].unique())
        print(f"  {label}: d_train values tested = {ds}")
    print("\n  ⚠ MISSING: HP universality check. None of task3/5/7 tested at d<600k.")
    print("     Locked LR/BS/aug/dropout may not hold at d=500 or d=30k.")
    print("     Task6 IS tested at both d=500 and d=600k — let us check if size winners differ:")
    if not df6.empty:
        df6 = df6.copy()
        df6["size_label"] = df6["hp"].apply(lambda h: h.get("size_label", "?"))
        for arch in df6["arch"].unique():
            sub = df6[df6["arch"] == arch]
            print(f"    {arch}:")
            for d in sorted(sub["d_train"].unique()):
                sub_d = sub[sub["d_train"] == d]
                agg = sub_d.groupby("size_label")["test_mse_at_best_val"].mean()
                best = agg.idxmin()
                print(f"      d={d}: best_size={best} ({agg[best]:.4f})  [all: {agg.to_dict()}]")


if __name__ == "__main__":
    main()
