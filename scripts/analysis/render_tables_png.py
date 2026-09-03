"""Render the inventory tables as paste-ready PNGs with FIXED column widths.

Why: pasting CSVs into Slides gives columns that fight each other -- long free-text cells force one
column wide and squeeze the rest, and adjusting one re-breaks the others. Rendering with explicit
per-column width fractions and hard text wrapping fixes the layout once, so the slide just holds an
image. A trimmed CSV is also written for anyone who needs it editable.
"""

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = os.path.expanduser("~/Downloads/pi_meeting_figs/editable_tables")
OUT = os.path.join(SRC, "rendered")
os.makedirs(OUT, exist_ok=True)

STATUS_COLORS = {
    "KEEP": "#dcfce7",
    "ADD": "#dbeafe",
    "ADD (JB)": "#c7d2fe",
    "REVISE": "#fef3c7",
    "PROPOSED": "#e9d5ff",
    "PARAMETERISE": "#fed7aa",
    "REMOVE": "#fecaca",
    "DEPRIORITISE": "#e5e7eb",
    "IMPLEMENTED": "#dcfce7",
    "NOT IMPLEMENTED": "#fecaca",
}


def render(df, name, widths, wraps, fontsize=7.2, row_h=0.055, title=None):
    """widths: fraction of total width per column (must sum ~1). wraps: chars per line per column."""
    wrapped, heights = [], []
    for _, row in df.iterrows():
        cells, lines = [], 1
        for val, w in zip(row.values, wraps):
            t = "" if pd.isna(val) else str(val)
            piece = textwrap.fill(t, w) if w else t
            lines = max(lines, piece.count("\n") + 1)
            cells.append(piece)
        wrapped.append(cells)
        heights.append(lines)

    fig_h = 0.55 + row_h * sum(heights) * 1.5
    fig, ax = plt.subplots(figsize=(13.5, max(2.2, fig_h)))
    ax.axis("off")
    tbl = ax.table(
        cellText=wrapped,
        colLabels=[textwrap.fill(c, 18) for c in df.columns],
        cellLoc="left",
        loc="upper left",
        colWidths=widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    status_col = next((i for i, c in enumerate(df.columns) if c.lower() == "status"), None)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("0.75")
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="w", weight="bold", fontsize=fontsize)
            cell.set_height(0.055)
        else:
            cell.set_height(row_h * heights[r - 1] * 1.35)
            if status_col is not None and c == status_col:
                cell.set_facecolor(STATUS_COLORS.get(str(df.iloc[r - 1, c]).strip(), "#ffffff"))
            elif r % 2 == 0:
                cell.set_facecolor("#f8fafc")
    if title:
        ax.set_title(title, fontsize=11, weight="bold", pad=6, loc="left")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


# ---- what-varies matrix: the single best overview slide
d = pd.read_csv(f"{SRC}/what_varies_matrix.csv")
render(
    d,
    "T1_what_varies",
    [0.24, 0.19, 0.19, 0.19, 0.19],
    [26, 20, 20, 20, 20],
    8.0,
    0.05,
    "Design space: what each strategy holds FIXED vs VARIES",
)

# ---- reservoir, split in two, with short columns only
res = pd.read_csv(f"{SRC}/reservoir_FULL.csv")
slim = res[
    [
        "Family",
        "Motif source (how identified)",
        "Background (how built)",
        "Placement rule",
        "Status",
    ]
]
render(
    slim.iloc[:9],
    "T2a_reservoir_part1",
    [0.16, 0.26, 0.24, 0.24, 0.10],
    [18, 30, 28, 28, 12],
    7.0,
    0.05,
    "Reservoir strategies (1 of 2) — natural, augmentation, our motif methods",
)
render(
    slim.iloc[9:],
    "T2b_reservoir_part2",
    [0.16, 0.26, 0.24, 0.24, 0.10],
    [18, 30, 28, 28, 12],
    7.0,
    0.05,
    "Reservoir strategies (2 of 2) — Shendure/JB family, variants, mixtures, removed",
)

# ---- acquisition / eval / hp
render(
    pd.read_csv(f"{SRC}/acquisition.csv"),
    "T3_acquisition",
    [0.20, 0.40, 0.18, 0.22],
    [22, 46, 18, 26],
    7.6,
    0.05,
    "Acquisition strategies",
)
render(
    pd.read_csv(f"{SRC}/eval_sets.csv"),
    "T4_eval_sets",
    [0.22, 0.32, 0.09, 0.16, 0.21],
    [24, 36, 10, 18, 24],
    7.6,
    0.05,
    "Evaluation sets",
)
render(
    pd.read_csv(f"{SRC}/hp_axes.csv"),
    "T5_hp_axes",
    [0.24, 0.30, 0.20, 0.26],
    [26, 34, 22, 30],
    7.6,
    0.05,
    "Hyperparameter axes",
)

# ---- trimmed CSVs (short cells) for anyone who wants them editable
slim.to_csv(f"{SRC}/reservoir_SLIM_for_slides.csv", index=False)
print(f"\nrendered PNGs -> {OUT}")
