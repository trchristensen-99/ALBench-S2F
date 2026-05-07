"""High-dimensional HP analysis: parallel-coords + edge detection.

For each architecture, plots all HP cells at once across LR / BS /
dropout / aug / size / D axes, colored by test_mse. Reveals:
- Whether low-mse cells cluster in specific HP regions (=> well-explored)
- Whether low-mse cells appear at axis edges (=> extend in that direction)
- Which axes show strong test_mse coupling vs which are noise

Generates:
- results/preflight/hp_heatmaps/parallel_coords_<arch>.png
- results/preflight/hp_heatmaps/edge_analysis.csv
- results/preflight/hp_heatmaps/edge_analysis_human.txt — human-readable summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "preflight" / "hp_heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results() -> pd.DataFrame:
    sources = [
        ("task3_lr_bs", "rev_complement"),
        ("task3b_lr_bs_d500", "rev_complement"),
        ("task3_retry_legnet_noaug", "none"),
        ("task3_retry_dream_attn_rcshift", "rc_shift"),
        ("task5_augmentations", None),
        ("task5_legnet_aug_confirm", None),
        ("task6_parameterization", None),
        ("task7_dropout", None),
        ("task7_dream_rnn_dropout_ext", None),
        ("task9_d_min_confirm", None),
        ("task_hp_universality", None),
        ("legnet_aug_crossover", None),
        ("legnet_rc_channel_test", None),
        ("hp_refinement_d30k", None),
    ]
    rows = []
    for src, default_aug in sources:
        d = REPO / "results" / "preflight" / src
        if not d.exists():
            continue
        for rj in d.rglob("result.json"):
            try:
                r = json.loads(rj.read_text())
                hp = r.get("hp", {}) or {}
                arch = r.get("arch")
                if arch == "legnet":
                    dropout = hp.get("dropout")
                elif arch == "dream_rnn":
                    dropout = hp.get("dropout_lstm")
                elif arch == "dream_attn":
                    dropout = hp.get("core_dropout")
                else:
                    dropout = None
                size_label = hp.get("size_label")
                if size_label is None:
                    parts = str(rj).split("/")
                    for p in parts:
                        if p.startswith("size_"):
                            size_label = p[len("size_") :]
                rows.append(
                    {
                        "source": src,
                        "arch": arch,
                        "d_train": r.get("d_train"),
                        "lr": hp.get("lr"),
                        "batch_size": hp.get("batch_size"),
                        "dropout": dropout,
                        "weight_decay": hp.get("weight_decay"),
                        "in_channels": hp.get("in_channels"),
                        "aug": r.get("augmentations") or default_aug,
                        "size_label": size_label or "default",
                        "n_params": r.get("n_params"),
                        "test_mse": r.get("test_mse_at_best_val"),
                        "best_val": r.get("best_val_mse"),
                        "best_epoch": r.get("best_epoch"),
                        "seed": r.get("seed"),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows)


def parallel_coords(df: pd.DataFrame, arch: str, out_png: Path):
    """Parallel-coordinates plot for one arch — all cells, colored by test_mse."""
    sub = df[df["arch"] == arch].dropna(subset=["test_mse"]).copy()
    if len(sub) < 5:
        return
    # Numeric encoding for categorical aug
    aug_order = ["none", "rev_complement", "rc_shift", "rc_shift_evoaug"]
    sub["aug_num"] = sub["aug"].map(
        lambda a: aug_order.index(a) if a in aug_order else len(aug_order)
    )
    sub["size_num"] = sub["size_label"].map({"half": 0, "default": 1, "double": 2}).fillna(1)
    sub["in_channels"] = sub["in_channels"].fillna(4)
    # Log10 for LR, BS, D
    sub["log_lr"] = np.log10(sub["lr"].astype(float))
    sub["log_bs"] = np.log10(sub["batch_size"].astype(float))
    sub["log_d"] = np.log10(sub["d_train"].astype(float))

    cols = ["log_lr", "log_bs", "dropout", "aug_num", "size_num", "in_channels", "log_d"]
    col_labels = ["log10(LR)", "log10(BS)", "dropout", "aug", "size", "in_channels", "log10(D)"]
    sub_clean = sub.dropna(subset=cols).copy()
    if len(sub_clean) < 5:
        return

    # Standardize each axis to [0,1] for plotting on shared axis
    norm = pd.DataFrame()
    for c in cols:
        v = sub_clean[c]
        norm[c] = (v - v.min()) / (v.max() - v.min() + 1e-9)

    # Colormap: lower MSE = better (use viridis_r)
    mse = sub_clean["test_mse"].to_numpy()
    mse_lo, mse_hi = np.quantile(mse, [0.05, 0.95])
    norm_mse = np.clip((mse - mse_lo) / (mse_hi - mse_lo + 1e-9), 0, 1)
    cmap = plt.get_cmap("viridis_r")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cols))
    # Sort by MSE so low-MSE lines drawn LAST (on top)
    order = np.argsort(-mse)
    for i in order:
        ax.plot(
            x,
            norm.iloc[i].values,
            color=cmap(norm_mse[i]),
            alpha=0.5 if mse[i] > np.median(mse) else 0.9,
            linewidth=0.6 if mse[i] > np.median(mse) else 1.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{arch}: parallel coords across HP axes (n={len(sub_clean)}; darker = better)")
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mse_lo, vmax=mse_hi))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("test_mse")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  Saved {out_png}")


def edge_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """For each (arch, d_train, aug), find the best cell + flag axes at edges
    of the explored grid."""
    rows = []
    for (arch, d, aug), sub in df.dropna(subset=["test_mse"]).groupby(
        ["arch", "d_train", "aug"], dropna=False
    ):
        if len(sub) < 2:
            continue
        best = sub.loc[sub["test_mse"].idxmin()]
        info = {
            "arch": arch,
            "d_train": int(d) if pd.notna(d) else None,
            "aug": aug,
            "n_cells": len(sub),
            "best_mse": float(best["test_mse"]),
        }
        for col in ["lr", "batch_size", "dropout"]:
            uniq = sorted(sub[col].dropna().unique())
            if len(uniq) < 2:
                continue
            best_v = best[col]
            if pd.isna(best_v):
                continue
            pos = uniq.index(best_v) if best_v in uniq else None
            if pos is None:
                continue
            n = len(uniq)
            if pos == 0:
                info[f"{col}_edge"] = f"LOWER ({best_v} of {uniq})"
            elif pos == n - 1:
                info[f"{col}_edge"] = f"UPPER ({best_v} of {uniq})"
            else:
                info[f"{col}_edge"] = "interior"
        rows.append(info)
    return pd.DataFrame(rows)


def main():
    df = load_all_results()
    print(f"Loaded {len(df):,} rows")
    print(f"  by arch: {df['arch'].value_counts().to_dict()}")
    print()

    # 1) Parallel coords per arch
    print("=== Parallel-coords plots ===")
    for arch in ["legnet", "dream_rnn", "dream_attn"]:
        parallel_coords(df, arch, OUT_DIR / f"parallel_coords_{arch}.png")

    # 2) Edge analysis
    print("\n=== Edge analysis (best cell per arch×D×aug) ===")
    edge_df = edge_analysis(df)
    edge_df = edge_df.sort_values("best_mse")
    edge_df.to_csv(OUT_DIR / "edge_analysis.csv", index=False)
    print(f"  saved {OUT_DIR / 'edge_analysis.csv'}")
    print()
    # Print human-readable
    lines = []
    lines.append(
        "EDGE ANALYSIS — for each (arch, d, aug), the best cell + which HPs are at grid edges"
    )
    lines.append("=" * 90)
    for _, r in edge_df.iterrows():
        edges = [
            f"{c}={r[f'{c}_edge']}"
            for c in ("lr", "batch_size", "dropout")
            if f"{c}_edge" in r and pd.notna(r[f"{c}_edge"])
        ]
        warn = " ⚠ EDGE" if any("LOWER" in e or "UPPER" in e for e in edges) else ""
        lines.append(
            f"  {r['arch']:<10} d={r['d_train']:>6}  aug={r['aug'] or '-':<16}  best_mse={r['best_mse']:.3f}  n={r['n_cells']}  | {', '.join(edges)}{warn}"
        )
    txt = "\n".join(lines)
    print(txt)
    (OUT_DIR / "edge_analysis_human.txt").write_text(txt + "\n")

    # 3) Promising-direction suggestions
    print("\n=== Promising directions to explore (where edges + low MSE coincide) ===")
    promising = []
    for _, r in edge_df.iterrows():
        if r["best_mse"] > 0.5:
            continue  # only promising if already low
        for col, label in [
            ("lr_edge", "LR"),
            ("batch_size_edge", "BS"),
            ("dropout_edge", "dropout"),
        ]:
            v = r.get(col)
            if pd.isna(v) or v == "interior":
                continue
            if "LOWER" in str(v):
                promising.append(
                    f"  {r['arch']:<10} d={r['d_train']:>6} aug={r['aug']:<16}: extend {label} BELOW {v.split('(')[1].split()[0]}"
                )
            elif "UPPER" in str(v):
                promising.append(
                    f"  {r['arch']:<10} d={r['d_train']:>6} aug={r['aug']:<16}: extend {label} ABOVE {v.split('(')[1].split()[0]}"
                )
    if not promising:
        print("  (none — all best cells are interior)")
    else:
        for p in promising:
            print(p)
        (OUT_DIR / "promising_extensions.txt").write_text("\n".join(promising) + "\n")


if __name__ == "__main__":
    main()
