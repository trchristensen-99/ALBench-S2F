"""Aggregate per-ckpt eval-set scores from Task 2 (or any preflight task)
into a single CSV + plot panel showing test_MSE vs D for each eval set.

Walks ``results/preflight/task2_d_min/<arch>/d<D>/seed<S>/eval_sets_panel.json``
and produces:
  - results/preflight/task2_eval_panel.csv (per arch × D × seed × panel)
  - results/preflight/figures/task2_eval_panels.png — panel-faceted scaling
    curve, log-x for D, log-y for MSE.

Usage:
  uv run --no-sync python scripts/preflight/aggregate_task2_eval_panels.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TASK2_DIR = REPO / "results" / "preflight" / "task2_d_min"
FIG_DIR = REPO / "results" / "preflight" / "figures"


def main():
    if not TASK2_DIR.exists():
        raise SystemExit(f"missing {TASK2_DIR}")
    rows = []
    for f in sorted(TASK2_DIR.rglob("eval_sets_panel.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        d_train = int(parts[-3].lstrip("d"))
        seed = int(parts[-2].replace("seed", ""))
        for panel_name, metrics in d.items():
            row = {
                "arch": arch,
                "d_train": d_train,
                "seed": seed,
                "panel": panel_name,
                "n_seqs": metrics.get("n_seqs"),
                "pearson_r": metrics.get("pearson_r"),
                "spearman_r": metrics.get("spearman_r"),
                "mse": metrics.get("mse"),
            }
            rows.append(row)
    if not rows:
        raise SystemExit("no eval_sets_panel.json files yet — submit score_task2_eval_sets.sh first")

    csv_path = REPO / "results" / "preflight" / "task2_eval_panel.csv"
    with csv_path.open("w") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["arch", "d_train", "seed", "panel", "n_seqs", "pearson_r", "spearman_r", "mse"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    # Plot: per-panel MSE vs D (log-log), per arch
    by_panel: dict[str, dict[tuple[str, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["mse"] is None:
            continue
        by_panel[r["panel"]][(r["arch"], r["d_train"])].append(r["mse"])

    panels = sorted(by_panel.keys())
    if not panels:
        return
    n_panels = len(panels)
    n_cols = 4
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = np.array(axes).flatten()
    archs = sorted({(a, _) for cells in by_panel.values() for (a, _) in cells})
    arch_set = sorted({a for a, _ in archs})
    colors = {"legnet": "tab:blue", "dream_rnn": "tab:orange", "dream_attn": "tab:green"}
    for idx, panel_name in enumerate(panels):
        ax = axes[idx]
        cells = by_panel[panel_name]
        for arch in arch_set:
            ds = sorted({d for (a, d) in cells if a == arch})
            means = []
            stds = []
            xs = []
            for d in ds:
                vals = cells.get((arch, d), [])
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                    xs.append(d)
            if xs:
                ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=arch, color=colors.get(arch))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(panel_name, fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel("test MSE (log)")
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel("D_train (log)")
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")
    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Task 2 D_min runs — test MSE vs D across eval-set panels (per arch, error bars across 3 seeds)")
    fig.tight_layout()
    out = FIG_DIR / "task2_eval_panels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
