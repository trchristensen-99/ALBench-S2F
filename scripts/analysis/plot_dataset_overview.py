#!/usr/bin/env python3
"""Generate Gosai dataset overview figure for poster.

4-panel figure:
  A: MPRA assay schematic description + key numbers
  B: Training distribution (log2FC histogram by data_project)
  C: Test sets comparison (in-dist, OOD, SNV distributions)
  D: Gosai vs Agarwal assay comparison (episomal vs lentiviral)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "poster_stowers"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    # Load data
    df = pd.read_csv(REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False)
    test_id = pd.read_csv(REPO / "data/k562/test_sets/test_chr7_13_all.tsv", sep="\t")
    test_ood = pd.read_csv(REPO / "data/k562/test_sets/test_ood_designed_k562.tsv", sep="\t")
    test_snv = pd.read_csv(REPO / "data/k562/test_sets/test_snv_pairs_hashfrag.tsv", sep="\t")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Dataset overview text ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")
    ax_a.set_title(
        "A. Gosai et al. 2024 — Episomal MPRA", fontsize=13, fontweight="bold", loc="left"
    )

    info = [
        ("Assay", "Episomal MPRA (pMPRAv3)"),
        ("Cell types", "K562, HepG2, SK-N-SH"),
        ("Library", "776,474 200-bp CRE oligos"),
        ("Sources", "UKBB GWAS (42%), GTEx eQTL (56%), CRE (2%)"),
        ("Readout", "log₂(RNA/DNA) fold-change"),
        ("", ""),
        ("Train", f"{len(df):,} seqs (all chr except 7,13,19,21,X)"),
        ("Val", "9,410 seqs (chr 19, 21, X)"),
        ("Test (ID)", f"{len(test_id):,} seqs (chr 7, 13)"),
        ("Test (OOD)", f"{len(test_ood):,} designed seqs (3 algorithms)"),
        ("Test (SNV)", f"{len(test_snv):,} variant pairs"),
    ]
    y = 9.5
    for label, value in info:
        if label:
            ax_a.text(
                0.5, y, label + ":", fontsize=10, fontweight="bold", va="top", family="monospace"
            )
            ax_a.text(3.5, y, value, fontsize=10, va="top")
        y -= 0.85

    # Pipeline boxes
    for x, w, label, color in [
        (0.5, 2.5, "Gosai\nMPRA Data", "#E8F5E9"),
        (3.5, 2.5, "AG S2\nOracle", "#E3F2FD"),
        (6.5, 3.0, "LegNet\nStudent", "#FFF3E0"),
    ]:
        rect = mpatches.FancyBboxPatch(
            (x, 0.2),
            w,
            1.3,
            boxstyle="round,pad=0.2",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_a.add_patch(rect)
        ax_a.text(x + w / 2, 0.85, label, fontsize=9, ha="center", va="center", fontweight="bold")

    # Arrows
    for x1, x2 in [(3.0, 3.5), (6.0, 6.5)]:
        ax_a.annotate("", xy=(x2, 0.85), xytext=(x1, 0.85), arrowprops=dict(arrowstyle="->", lw=2))

    # ── Panel B: Training distribution ──────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    colors_proj = {"UKBB": "#e74c3c", "GTEX": "#3498db", "CRE": "#2ecc71"}
    for proj in ["GTEX", "UKBB", "CRE"]:
        sub = df[df["data_project"] == proj]
        ax_b.hist(
            sub["K562_log2FC"].clip(-5, 8),
            bins=100,
            alpha=0.6,
            label=f"{proj} (n={len(sub):,})",
            color=colors_proj[proj],
            density=True,
        )

    ax_b.set_xlabel("K562 log₂FC", fontsize=11)
    ax_b.set_ylabel("Density", fontsize=11)
    ax_b.set_title(
        "B. Training Data Distribution by Source", fontsize=13, fontweight="bold", loc="left"
    )
    ax_b.legend(fontsize=9)
    ax_b.set_xlim(-5, 8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ── Panel C: Test set distributions ─────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    ax_c.hist(
        test_id["K562_log2FC"].clip(-5, 8),
        bins=80,
        alpha=0.6,
        label=f"In-dist chr7/13 (n={len(test_id):,})",
        color="#3498db",
        density=True,
    )
    ax_c.hist(
        test_ood["K562_log2FC"].clip(-5, 10),
        bins=80,
        alpha=0.6,
        label=f"OOD designed (n={len(test_ood):,})",
        color="#e74c3c",
        density=True,
    )

    # SNV: show ref distribution
    ax_c.hist(
        test_snv["K562_log2FC_ref"].clip(-5, 8),
        bins=80,
        alpha=0.4,
        label=f"SNV ref (n={len(test_snv):,})",
        color="#2ecc71",
        density=True,
    )

    ax_c.set_xlabel("K562 log₂FC", fontsize=11)
    ax_c.set_ylabel("Density", fontsize=11)
    ax_c.set_title("C. Test Set Distributions", fontsize=13, fontweight="bold", loc="left")
    ax_c.legend(fontsize=9)
    ax_c.set_xlim(-5, 10)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ── Panel D: OOD design methods ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    method_colors = {
        "FastSeqProp": "#9b59b6",
        "Simulated_Annealing": "#e67e22",
        "AdaLead": "#1abc9c",
    }
    for method in ["FastSeqProp", "Simulated_Annealing", "AdaLead"]:
        sub = test_ood[test_ood["method"] == method]
        label = method.replace("_", " ")
        ax_d.hist(
            sub["K562_log2FC"].clip(-2, 10),
            bins=60,
            alpha=0.6,
            label=f"{label} (n={len(sub):,})",
            color=method_colors[method],
            density=True,
        )

    ax_d.axvline(
        x=df["K562_log2FC"].median(),
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Train median ({df['K562_log2FC'].median():.2f})",
    )
    ax_d.set_xlabel("K562 log₂FC", fontsize=11)
    ax_d.set_ylabel("Density", fontsize=11)
    ax_d.set_title(
        "D. OOD Test: Computationally Designed CREs", fontsize=13, fontweight="bold", loc="left"
    )
    ax_d.legend(fontsize=9)
    ax_d.set_xlim(-2, 10)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    fig.savefig(OUT / "panel1_dataset_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "panel1_dataset_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: panel1_dataset_overview.png")

    # ── Bonus: Gosai vs Agarwal comparison (clean version) ──────────
    # Only if Agarwal data exists
    agarwal_path = REPO / "data" / "external" / "agarwal_2023"
    if not agarwal_path.exists():
        print("Agarwal data not found, skipping comparison panel")
        return

    # Use the existing comparison figure instead
    existing = REPO / "results" / "dataset_comparison" / "agarwal_gosai_comparison.png"
    if existing.exists():
        import shutil

        shutil.copy2(existing, OUT / "extra_gosai_vs_agarwal.png")
        print(f"Copied existing Gosai vs Agarwal comparison")


if __name__ == "__main__":
    main()
