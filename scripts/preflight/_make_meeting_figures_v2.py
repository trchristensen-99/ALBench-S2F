"""V2 meeting figures using PROPER 10-fold ENSEMBLED metrics + bias-vs-target figure.

Key corrections vs v1:
- Use ENSEMBLED predictions (averaged across 10 folds), not per-fold averages.
  Per-fold mean understates ensemble performance by ~0.005-0.04.
- Add explicit bias-vs-expected figure showing how each oracle's predictions
  on negative control sequences compare to ground-truth distributions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
OUT = REPO / "results/preflight/figures/meeting"
OUT.mkdir(parents=True, exist_ok=True)


def fig02_v2_decision_with_ensemble(out):
    """Compare per-fold mean vs ENSEMBLED metric across oracles."""
    ens = json.loads((REPO / "results/preflight/oracle_ensemble_metrics.json").read_text())
    oracles = ["baseline", "c28_10fold", "c63_10fold", "c86_10fold", "c91_10fold"]
    labels_map = {
        "baseline": "baseline",
        "c28_10fold": "c28 (dinuc)",
        "c63_10fold": "c63 (Sahu+cpg_inv)",
        "c86_10fold": "c86 (blk012+Sahu)",
        "c91_10fold": "c91 (blk012+dinuc+cpg_inv) — WINNER",
    }
    colors = ["gray", "steelblue", "lightsteelblue", "lightcoral", "tomato"]
    metrics = [
        ("in_dist", "Test ID (chr 7+13) Pearson R"),
        ("ood", "OOD Pearson R"),
        ("snv_delta", "SNV delta Pearson R"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for mi, (mkey, label) in enumerate(metrics):
        ax = axes[mi]
        per_fold = [ens[o][f"{mkey}_per_fold_R_mean"] for o in oracles]
        per_fold_std = [ens[o][f"{mkey}_per_fold_R_std"] for o in oracles]
        ensemble = [ens[o][f"{mkey}_ensemble_R"] for o in oracles]
        x = np.arange(len(oracles))
        ax.bar(x - 0.2, per_fold, 0.4, color=colors, alpha=0.5, edgecolor="black",
               yerr=per_fold_std, capsize=3, label="per-fold mean")
        ax.bar(x + 0.2, ensemble, 0.4, color=colors, edgecolor="black",
               label="10-fold ENSEMBLE")
        ax.set_xticks(x)
        ax.set_xticklabels([labels_map[o] for o in oracles], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        for xi, e in zip(x, ensemble):
            ax.text(xi + 0.2, e + 0.002, f"{e:.3f}", ha="center", fontsize=7, fontweight="bold")
        for xi, p in zip(x, per_fold):
            ax.text(xi - 0.2, p - 0.015, f"{p:.3f}", ha="center", fontsize=7)
        if mi == 0:
            ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        # Set ylim near data range for visibility
        all_vals = per_fold + ensemble
        ax.set_ylim(min(all_vals) - 0.02, max(all_vals) + 0.01)
    fig.suptitle("Per-fold vs ENSEMBLED metrics — Baseline ≈ 0.94 ID, c91 ≈ 0.96 ID",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig08_bias_vs_target(out):
    """Show each oracle's predicted mean on negative controls vs expected ground truth."""
    summary = json.loads((REPO / "results/preflight/bias_eval_summary.json").read_text())
    # Expected ground truth (from Gosai ctrl_neg measured)
    GOSAI_CTRL_NEG_MEAN = 0.266
    GOSAI_CTRL_NEG_STD = 0.491
    EXPECTED = {
        "random_dna": 0.0,      # uniform random expected to have no signal
        "shuffled": GOSAI_CTRL_NEG_MEAN,  # dinuc-shuffled real, like Gosai ctrl_neg
        "intergenic": 0.0,       # real intergenic, basal expression
    }
    LABELS = {
        "random_dna": f"Random DNA\n(target ~0.0)",
        "shuffled": f"Dinuc-shuffled\n(target +{GOSAI_CTRL_NEG_MEAN:.2f}\nfrom Gosai ctrl_neg)",
        "intergenic": f"Intergenic\n(target ~0.0)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    oracles = list(summary.keys())
    colors = {"c28_10fold": "steelblue", "c63_10fold": "lightsteelblue",
              "c86_10fold": "lightcoral", "c91_10fold": "tomato"}
    for ci, cat in enumerate(["random_dna", "shuffled", "intergenic"]):
        ax = axes[0][ci]
        means = [summary[o][cat]["mean"] for o in oracles]
        stds = [summary[o][cat]["std"] for o in oracles]
        bars = ax.bar(range(len(oracles)), means,
                      color=[colors.get(o, "gray") for o in oracles],
                      yerr=stds, capsize=5, edgecolor="black", alpha=0.8)
        # Expected line
        target = EXPECTED[cat]
        ax.axhline(target, color="red", linestyle="--", linewidth=2,
                   label=f"target = {target:+.2f}")
        if cat == "shuffled":
            ax.axhspan(target - GOSAI_CTRL_NEG_STD, target + GOSAI_CTRL_NEG_STD,
                       color="red", alpha=0.1, label=f"±1σ Gosai ctrl_neg")
        ax.set_xticks(range(len(oracles)))
        ax.set_xticklabels([o.replace("_10fold", "") for o in oracles], rotation=30, ha="right",
                           fontsize=8)
        ax.set_ylabel("Predicted mean", fontsize=10)
        ax.set_title(LABELS[cat], fontsize=10)
        for x, m in enumerate(means):
            ax.text(x, m + max(stds)*0.15, f"{m:+.2f}", ha="center", fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "Oracle predictions on negative control sequences vs expected ground truth",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def fig09_bias_composite(out):
    """Composite bias score = mean |distance from expected| across categories."""
    summary = json.loads((REPO / "results/preflight/bias_eval_summary.json").read_text())
    oracles = list(summary.keys())
    rows = []
    for o in oracles:
        r = {
            "oracle": o,
            "random_err": abs(summary[o]["random_dna"]["mean"] - 0.0),
            "shuffled_err": abs(summary[o]["shuffled"]["mean"] - 0.266),
            "intergenic_err": abs(summary[o]["intergenic"]["mean"] - 0.0),
        }
        r["composite"] = (r["random_err"] + r["shuffled_err"] + r["intergenic_err"]) / 3
        rows.append(r)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.22
    cats = ["random_err", "shuffled_err", "intergenic_err", "composite"]
    cat_labels = ["|random - 0|", "|shuffled - 0.27|", "|intergenic - 0|", "composite avg"]
    colors_cats = ["#4c72b0", "#dd8452", "#55a868", "black"]
    for i, (c, l, col) in enumerate(zip(cats, cat_labels, colors_cats)):
        ax.bar(x + (i - 1.5) * w, df[c], w, label=l, color=col, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df.oracle.str.replace("_10fold", ""), rotation=30, ha="right")
    ax.set_ylabel("Absolute bias (lower = better calibrated)")
    ax.set_title("Negative-control calibration error per oracle (lower is better)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    fig02_v2_decision_with_ensemble(OUT / "02_v2_decision_ensembled.png")
    fig08_bias_vs_target(OUT / "08_bias_vs_target.png")
    fig09_bias_composite(OUT / "09_bias_composite.png")
    print(f"\nNew figures in {OUT}")


if __name__ == "__main__":
    main()
