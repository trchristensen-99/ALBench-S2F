#!/usr/bin/env python
"""Plot the from-scratch reservoir scaling curves and emit a paste-ready table."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "random": "#888888",
    "motif_syntax_core": "#1b7837",
    "motif_ct_enriched": "#5aae61",
    "evoaug": "#2166ac",
    "mutagenesis": "#b2182b",
}
LABEL = {
    "random": "random (floor)",
    "motif_syntax_core": "motif: syntax core",
    "motif_ct_enriched": "motif: CT-enriched",
    "evoaug": "EvoAug",
    "mutagenesis": "PRM (mutagenesis)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default="outputs/analysis/stage4_curves.json")
    ap.add_argument("--out", default="outputs/analysis/stage4_curves.png")
    ap.add_argument(
        "--min-models",
        type=int,
        default=3,
        help="hide points built from fewer models than this; with n<3 the "
        "greedy+ElasticNet selection has nothing to select FROM, so the "
        "point reflects search budget rather than the reservoir",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.curves).read_text())
    by_arm: dict[str, list] = defaultdict(list)
    for cell in data.values():
        by_arm[cell["arm"]].append(cell)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    rows = []
    for arm, cells in sorted(by_arm.items()):
        cells = sorted(cells, key=lambda c: c["D"])
        keep = [c for c in cells if c["n_models"] >= args.min_models]
        thin = [c for c in cells if c["n_models"] < args.min_models]
        col = COLORS.get(arm, "#333333")
        if keep:
            axes[0].plot(
                [c["D"] for c in keep],
                [c["ensemble_r"] for c in keep],
                "o-",
                color=col,
                lw=2.5,
                ms=8,
                label=LABEL.get(arm, arm),
            )
        if thin:
            axes[0].plot(
                [c["D"] for c in thin],
                [c["ensemble_r"] for c in thin],
                "o",
                color=col,
                ms=7,
                mfc="none",
                alpha=0.6,
            )
        for c in cells:
            axes[1].plot(
                c["D"],
                c["ensemble_r"] - c["best_single_r_by_val"],
                "o",
                color=col,
                ms=9 if c["n_models"] >= args.min_models else 6,
                mfc=col if c["n_models"] >= args.min_models else "none",
            )
            rows.append(
                (
                    arm,
                    c["D"],
                    c["n_models"],
                    c["ensemble_r"],
                    c["best_single_r_by_val"],
                    c["ensemble_r"] - c["best_single_r_by_val"],
                )
            )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("sequences added (from scratch, base = 0)")
    axes[0].set_ylabel("ensemble Pearson r (held-out genomic)")
    axes[0].set_title("Reservoir scaling, N=5 ensemble")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].set_xscale("log")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xlabel("sequences added")
    axes[1].set_ylabel("ensemble − best single model")
    axes[1].set_title("What ensembling buys")
    axes[1].grid(alpha=0.3)

    fig.text(
        0.5,
        0.005,
        "open markers: fewer than %d models searched — budget-limited, not comparable"
        % args.min_models,
        ha="center",
        fontsize=12,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")

    tsv = Path(args.out).with_suffix(".tsv")
    with tsv.open("w") as fh:
        fh.write("reservoir\tD\tn_models\tensemble_r\tbest_single_r\tensemble_gain\n")
        for r in sorted(rows, key=lambda x: (x[0], x[1])):
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.4f}\t{r[5]:+.4f}\n")
    print(f"wrote {args.out}, .pdf, and {tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
