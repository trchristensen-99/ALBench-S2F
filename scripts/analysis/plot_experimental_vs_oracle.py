"""Experimental (real K562 log2FC) vs oracle (AG_S2 chr-split natural ensemble)
activity distributions for the 5 test sets where we have experimental measurements:
  Genomic Reference, SNV Ref, SNV Effect (Δ), High-Activity Designed (OOD), ctrl_neg.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
TEST_DIR = REPO / "data/k562/test_sets_ag_s2_chrsplit"
OUT = REPO / "outputs/experimental_vs_oracle_distributions"


def load_panels():
    panels = {}
    # Genomic Reference
    z = np.load(TEST_DIR / "genomic_oracle.npz", allow_pickle=True)
    panels["genomic"] = {
        "label": "Genomic Reference (chr 7+13)",
        "experimental": z["true_label"].astype(np.float32),
        "oracle": z["oracle_mean"].astype(np.float32),
    }
    # SNV Ref
    z = np.load(TEST_DIR / "snv_oracle.npz", allow_pickle=True)
    true_ref = z["true_alt_label"].astype(np.float32) - z["true_delta"].astype(np.float32)
    panels["snv_ref"] = {
        "label": "SNV Ref",
        "experimental": true_ref,
        "oracle": z["ref_mean"].astype(np.float32),
    }
    # SNV Effect (Δ)
    panels["snv_delta"] = {
        "label": "SNV Effect (Δ log2FC)",
        "experimental": z["true_delta"].astype(np.float32),
        "oracle": z["delta_mean"].astype(np.float32),
    }
    # OOD
    z = np.load(TEST_DIR / "ood_oracle.npz", allow_pickle=True)
    panels["ood"] = {
        "label": "High-Activity Designed",
        "experimental": z["true_label"].astype(np.float32),
        "oracle": z["oracle_mean"].astype(np.float32),
    }
    # ctrl_neg (negative control intergenic sequences from Gosai)
    ctrl_path = TEST_DIR / "ctrl_neg_oracle.npz"
    if ctrl_path.exists():
        z = np.load(ctrl_path, allow_pickle=True)
        panels["ctrl_neg"] = {
            "label": "ctrl_neg (intergenic)",
            "experimental": z["true_label"].astype(np.float32),
            "oracle": z["oracle_mean"].astype(np.float32),
        }
    else:
        print(
            f"[WARN] {ctrl_path} not found — ctrl_neg panel will be skipped. "
            "Run scripts/score_ctrl_neg_ag_s2.py once the chr-split AG_S2 oracle is ready."
        )
        panels["ctrl_neg"] = None
    return panels


def finite(x):
    if x is None:
        return None
    return x[np.isfinite(x)]


def plot_panel(ax, data, title, x_range):
    if data is None:
        ax.text(
            0.5,
            0.5,
            "ctrl_neg oracle\nnot yet computed\n(awaiting chr-split AG retrain)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            style="italic",
            color="#666666",
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    bins = np.linspace(x_range[0], x_range[1], 80)
    sources = [
        ("experimental", "Experimental (real K562 log2FC)", "#000000", "-"),
        ("oracle", "Oracle (AG_S2 chr-split ensemble)", "#0072B2", "-"),
    ]
    for key, label, color, ls in sources:
        y = finite(data.get(key))
        if y is None or len(y) == 0:
            continue
        hist, edges = np.histogram(y, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers,
            hist,
            color=color,
            linestyle=ls,
            label=f"{label}\n  μ={y.mean():.2f}, σ={y.std():.2f}, n={len(y):,}",
            linewidth=2.0,
            alpha=0.90,
        )
    ax.set_xlabel("log2FC", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(*x_range)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = load_panels()
    print("=== loaded panels ===")
    for k, d in panels.items():
        if d is None:
            print(f"  {k}: MISSING")
            continue
        for sk in ["experimental", "oracle"]:
            arr = finite(d.get(sk))
            n = len(arr) if arr is not None else 0
            print(f"  {k}/{sk}: n={n}")

    # Panel-specific x-ranges
    ranges = {
        "genomic": (-3, 8),
        "snv_ref": (-3, 8),
        "snv_delta": (-2, 2),
        "ood": (-3, 10),
        "ctrl_neg": (-3, 6),
    }

    panel_order = [
        ("genomic", "Genomic Reference (chr 7+13)"),
        ("snv_ref", "SNV Ref"),
        ("snv_delta", "SNV Effect (Δ log2FC)"),
        ("ood", "High-Activity Designed"),
        ("ctrl_neg", "ctrl_neg (negative control)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    for ax, (key, title) in zip(axes, panel_order):
        plot_panel(ax, panels[key], title, ranges[key])
    fig.tight_layout()
    fig.savefig(OUT / "experimental_vs_oracle.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "experimental_vs_oracle.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved 5-panel comparison")

    print("\n=== Moment table ===")
    print(f"  {'panel':<22}  {'source':<22}  {'n':>8}  {'mean':>7}  {'std':>6}")
    for key, d in panels.items():
        if d is None:
            continue
        for sk in ["experimental", "oracle"]:
            arr = finite(d.get(sk))
            if arr is None or len(arr) == 0:
                continue
            print(f"  {key:<22}  {sk:<22}  {len(arr):>8,}  {arr.mean():>7.3f}  {arr.std():>6.3f}")


if __name__ == "__main__":
    main()
