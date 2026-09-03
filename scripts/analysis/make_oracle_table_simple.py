"""Presentation table: one honest number per evaluation set, next to the deployed ensemble.

Two column groups, each reported as r and MSE:
  single model, out-of-fold   each sequence scored only by the fold that held it out.
  ensemble of 10              the deployed oracle. Not held out - every sequence was in 9 of the
                              10 folds' training data.

r and MSE are both shown because they can disagree: r is scale-free and judges ranking, MSE judges
the values themselves. That matters when the predictions are used as training targets.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HDR = "#1e293b"
BAND = "#f4f6f9"
RULE = "#94a3b8"

# (label, n, oof_r, oof_mse, ens_r, ens_mse)
ROWS = [
    ("Genomic reference", "30,659", "0.95", "0.13", "0.97", "0.07"),
    ("Designed high-activity", "22,962", "0.84", "0.76", "0.98", "0.10"),
    ("Negative controls", "471", "0.82", "0.08", "0.93", "0.03"),
    ("SNV alleles (absolute)", "56,144", "0.92", "0.17", "0.96", "0.09"),
    ("RULE",),
    ("SNV effect (alt − ref)", "29,493", "0.39", "0.19", "0.46", "0.17"),
]
CAPTION = (
    "Oracle pseudo-labels vs measured K562 activity.  Out-of-fold = each sequence scored only by the fold\n"
    "that held it out.  The ensemble is the deployed 10-fold mean and is not held out: every sequence was in\n"
    "9 of the 10 folds' training data.  MSE is in log2FC units, the same scale as the labels."
)
XS = [0.0, 0.395, 0.525, 0.655, 0.79, 0.90, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=os.path.expanduser("~/Downloads/notion_updates/fig_oracle_simple.png")
    )
    ap.add_argument("--title", default="Oracle label quality")
    ap.add_argument("--dpi", type=int, default=240)
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.8, 3.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rh = 0.112
    top = 0.80
    ax.text(0, 0.975, a.title, fontsize=16, fontweight="bold", va="top", color="#0f172a")

    # two-tier header: group labels on top, metric names beneath
    gh = rh * 0.72
    ax.add_patch(Rectangle((0, top - rh - gh), 1, rh + gh, facecolor=HDR, edgecolor="none"))
    ax.text(
        (XS[2] + XS[4]) / 2,
        top - gh / 2,
        "single model, out-of-fold",
        fontsize=10,
        color="white",
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        (XS[4] + XS[6]) / 2,
        top - gh / 2,
        "ensemble of 10",
        fontsize=10,
        color="#aebdcd",
        ha="center",
        va="center",
    )
    for i, c in enumerate(["evaluation set", "n", "r", "MSE", "r", "MSE"]):
        al = "left" if i == 0 else "right"
        x = XS[i] + 0.010 if al == "left" else XS[i + 1] - 0.010
        ax.text(
            x,
            top - gh - rh / 2,
            c,
            fontsize=11.5,
            color="white",
            fontweight="bold",
            ha=al,
            va="center",
        )

    y = top - rh - gh
    band = 0
    for row in ROWS:
        if row[0] == "RULE":
            ax.plot([0, 1], [y, y], color=RULE, lw=1.0)
            band = 0
            continue
        y -= rh
        if band % 2 == 1:
            ax.add_patch(Rectangle((0, y), 1, rh, facecolor=BAND, edgecolor="none"))
        band += 1
        for i, v in enumerate(row):
            al = "left" if i == 0 else "right"
            x = XS[i] + 0.010 if al == "left" else XS[i + 1] - 0.010
            bold = i in (2, 3)  # the out-of-fold pair is the headline
            col = "#475569" if i == 1 else ("#64748b" if i >= 4 else "#0f172a")
            ax.text(
                x,
                y + rh / 2,
                v,
                fontsize=12 if i < 2 else 12.5,
                ha=al,
                va="center",
                color=col,
                fontweight="bold" if bold else "normal",
            )

    ax.plot([0, 1], [y, y], color=HDR, lw=1.4)
    ax.text(0, y - 0.042, CAPTION, fontsize=8.8, color="#475569", va="top", linespacing=1.55)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
