"""Render the oracle evaluation table as a PNG, for pasting into Notion and Slack.

A space-padded text table only survives inside a code block, and even then Slack and Notion render
it at different widths. An image renders identically everywhere, so this is the version to share.
Numbers live in ROWS below - edit there and re-run.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HDR = "#1e293b"
BAND = "#f1f5f9"
GRP = "#e2e8f0"
GOOD = "#15803d"
BAD = "#b91c1c"
MUT = "#64748b"

COLS = ["eval set", "n", "all-folds", "out-of-fold", "inflation", "ceiling", "of ceiling"]
WIDTHS = [0.30, 0.10, 0.12, 0.13, 0.12, 0.11, 0.12]

# (label, n, all_folds, out_of_fold, inflation, ceiling, frac)  — None renders as an em dash
ROWS = [
    ("GROUP", "Absolute activity"),
    ("Genomic reference", "30,659", "0.9745", "0.9496", "-0.025", "0.975", "97%"),
    ("Designed high-activity", "22,962", "0.9815", "0.8398", "-0.142", "0.964", "87%"),
    ("Negative controls", "471", "0.9274", "0.8231", "-0.104", "0.915", "90%"),
    ("SNV ref allele", "29,383", "0.9561", "0.9212", "-0.035", None, None),
    ("SNV alt allele", "26,761", "0.9578", "0.9238", "-0.034", None, None),
    ("GROUP", "Variant effect (alt - ref)"),
    ("Monoallelic", "29,493", "0.4621", "0.3928", "-0.069", "~0.61", "64%"),
    ("Multiallelic", "1,714", "0.6638", "0.6183", "-0.046", "~0.70", "88%"),
]

FOOT = (
    "all-folds = deployed 10-fold ensemble mean.   out-of-fold = each sequence scored only by the fold that held it out.\n"
    "Folds are a random split of the 856,252-sequence training pool with no chromosome exclusion, so every eval sequence\n"
    "was in 9 of 10 folds' training data.  A variant effect needs both alleles scored by the same fold (~1/10 of pairs),\n"
    "hence the smaller n.  Monoallelic variants were assayed in one oligo context, multiallelic in several; multiallelic\n"
    "variants were selected for large effects and rank more easily.  SNV rows use true single-nucleotide substitutions\n"
    "only.  Ceilings are tentative, from per-oligo standard errors."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=os.path.expanduser("~/Downloads/notion_updates/fig_oracle_performance.png")
    )
    ap.add_argument("--title", default="Oracle performance vs measured K562 activity")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument(
        "--no_ceiling",
        action="store_true",
        help="drop the ceiling columns. The Slack version omits them, since the ceiling "
        "derivation needs explaining and the inflation columns stand on their own.",
    )
    args = ap.parse_args()

    cols, widths = list(COLS), list(WIDTHS)
    rows = [list(r) for r in ROWS]
    foot = FOOT
    if args.no_ceiling:
        cols = cols[:5]
        widths = [0.32, 0.12, 0.15, 0.16, 0.14]
        rows = [r if r[0] == "GROUP" else r[:5] for r in rows]
        foot = (
            "\n".join(l for l in FOOT.split("\n") if "Ceilings" not in l).rstrip().rstrip(".") + "."
        )
    globals()["COLS_R"], globals()["WIDTHS_R"] = cols, widths

    rh = 0.058
    n_disp = len(rows)
    fig_h = 1.05 + n_disp * rh * 4.6
    fig, ax = plt.subplots(figsize=(9.4 if args.no_ceiling else 11.6, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x0 = [0.0]
    for w in widths:
        x0.append(x0[-1] + w)
    total = x0[-1]
    x0 = [x / total for x in x0]

    top = 0.895
    ax.text(0, 0.975, args.title, fontsize=15, fontweight="bold", va="top", color="#0f172a")

    # header
    ax.add_patch(Rectangle((0, top - rh), 1, rh, facecolor=HDR, edgecolor="none"))
    for i, c in enumerate(cols):
        align = "left" if i == 0 else "right"
        xx = x0[i] + 0.008 if i == 0 else x0[i + 1] - 0.008
        ax.text(
            xx,
            top - rh / 2,
            c,
            fontsize=10.5,
            color="white",
            fontweight="bold",
            ha=align,
            va="center",
        )

    y = top - rh
    band = 0
    for row in rows:
        if row[0] == "GROUP":
            y -= rh * 0.92
            ax.add_patch(Rectangle((0, y), 1, rh * 0.92, facecolor=GRP, edgecolor="none"))
            ax.text(
                x0[0] + 0.008,
                y + rh * 0.46,
                row[1],
                fontsize=10.5,
                fontweight="bold",
                color="#334155",
                ha="left",
                va="center",
            )
            band = 0
            continue
        y -= rh
        if band % 2 == 1:
            ax.add_patch(Rectangle((0, y), 1, rh, facecolor=BAND, edgecolor="none"))
        band += 1
        for i, val in enumerate(row):
            txt = "—" if val is None else str(val)
            color, weight = "#0f172a", "normal"
            if i == 4 and val not in (None, "PENDING"):
                color, weight = BAD, "bold"
            if i == 3 and val not in (None, "PENDING"):
                weight = "bold"
            if txt == "PENDING":
                color, weight = MUT, "normal"
            if i == 6 and val not in (None, "PENDING"):
                color = GOOD if int(str(val).rstrip("%")) >= 95 else "#0f172a"
            align = "left" if i == 0 else "right"
            xx = x0[i] + 0.008 if i == 0 else x0[i + 1] - 0.008
            ax.text(
                xx,
                y + rh / 2,
                txt,
                fontsize=10.5,
                color=color,
                fontweight=weight,
                ha=align,
                va="center",
                family="DejaVu Sans",
            )
    ax.plot([0, 1], [y, y], color=HDR, lw=1.2)
    ax.text(0, y - 0.028, foot, fontsize=7.6, color="#475569", va="top", linespacing=1.5)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
