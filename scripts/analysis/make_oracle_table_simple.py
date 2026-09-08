"""Presentation table: oracle accuracy on HELD-OUT TEST FOLDS, individual models vs an ensemble.

Design choices, each answering a specific request:
  * every number is a test-fold number, never the val fold that early stopping selected on
  * individual models are shown as a mean over the 10 folds with the spread across folds, since
    with a rotating split the average across folds is the only meaningful summary
  * an 8-model within-fold ensemble is shown alongside, which is the honest analogue of ensembling
  * both r and MSE, plus the LABEL SD, because MSE is not comparable across evaluation sets with
    different dynamic ranges - SNV effect spans 0.47 against 1.2-2.0 for the activity sets
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

# (label, n/fold, label SD, individual r, individual MSE, ens r, ens MSE)
ROWS = [
    ("WT / genomic ref", "39,143", "1.19", "0.915 ± .008", "0.230", "0.918", "0.219"),
    ("SNV alt allele", "39,169", "1.18", "0.916 ± .007", "0.227", "0.919", "0.212"),
    ("Designed high-activity", "2,296", "1.59", "0.876 ± .005", "0.599", "0.892", "0.530"),
    ("Negative controls", "86", "1.98", "0.966 ± .015", "0.364", "0.985", "0.165"),
    ("RULE",),
    ("SNV effect (alt − ref)", "35,691", "0.47", "0.402 ± .023", "0.186", "0.404", "0.193"),
]
HEADS = ["evaluation set", "n / fold", "label SD", "r", "MSE", "r", "MSE"]
XS = [0.0, 0.265, 0.375, 0.475, 0.635, 0.755, 0.875, 1.0]
CAPTION = (
    "All numbers are on a HELD-OUT TEST FOLD, never the fold early stopping selected on.  Folds are 2-3\n"
    "chromosomes of ref/alt sequences plus a random tenth of the designed sequences; for any fold, 8 of the 10\n"
    "models trained on it, 1 validated on it, 1 tested on it.  Individual = mean over the 10 models, each on its\n"
    "own test fold, ± spread across folds.  8-model = 8 seeds on one fold's split, averaged.  Label SD is given\n"
    "because MSE is not comparable across sets with different dynamic ranges: SNV effect spans 0.47 against\n"
    "1.2-2.0 for the activity sets, so its low MSE reflects small targets rather than better prediction.\n"
    "Training: full encoder unfrozen, reverse-complement and native-context shift augmentation."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser(
        "~/Downloads/notion_updates/fig_oracle_simple.png"))
    ap.add_argument("--title", default="Oracle label quality (held-out test folds)")
    ap.add_argument("--dpi", type=int, default=240)
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rh = 0.098
    top = 0.80
    ax.text(0, 0.975, a.title, fontsize=16, fontweight="bold", va="top", color="#0f172a")

    gh = rh * 0.72
    ax.add_patch(Rectangle((0, top - rh - gh), 1, rh + gh, facecolor=HDR, edgecolor="none"))
    ax.text((XS[3] + XS[5]) / 2, top - gh / 2, "individual model",
            fontsize=10.5, color="white", fontweight="bold", ha="center", va="center")
    ax.text((XS[5] + XS[7]) / 2, top - gh / 2, "8-model ensemble",
            fontsize=10.5, color="#aebdcd", ha="center", va="center")
    for i, c in enumerate(HEADS):
        al = "left" if i == 0 else "right"
        x = XS[i] + 0.010 if al == "left" else XS[i + 1] - 0.010
        ax.text(x, top - gh - rh / 2, c, fontsize=11.5, color="white",
                fontweight="bold", ha=al, va="center")

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
            bold = i in (3, 4)                      # individual model is the headline pair
            col = "#475569" if i in (1, 2) else ("#64748b" if i >= 5 else "#0f172a")
            ax.text(x, y + rh / 2, v, fontsize=11.5 if i < 3 else 12, ha=al, va="center",
                    color=col, fontweight="bold" if bold else "normal")

    ax.plot([0, 1], [y, y], color=HDR, lw=1.4)
    ax.text(0, y - 0.038, CAPTION, fontsize=8.2, color="#475569", va="top", linespacing=1.5)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
