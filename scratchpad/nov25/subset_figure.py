"""fig21: not just WHICH reservoir a rule takes from, but which sequences inside it.

The collaborators' question after the Nov 25 meeting was one level finer than fig18.
fig18 says uncertainty takes 40.5% of its batch from motif_shared_core. It does not say
whether it takes the sparse plantings or the dense ones -- and for a 4M order that is the
difference between "buy more motif sequences" and "buy motif sequences built this way".

Left panel is the reservoir level, redrawn here beside the right panel so the two are read
on one axis. Right panel is inside motif_shared_core, by how many sites were planted.

Both are drawn as a RATIO against the share the pool offered, with 1.0 marked. A raw
percentage is not comparable between the panels (4 reservoirs give a 0.25 baseline, 5 site
counts give 0.20), and the ratio is also what the control has to land on: `random` is here
precisely so a reader can see the measurement return 1.0 when nothing is being selected
for.

    python scratchpad/nov25/subset_figure.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
GREY = "#6b6b6b"

# The three arms share one 4-way pool (random / motif_shared_core / evoaug / genomic,
# 75,000 candidates each), so the only thing that differs between them is the rule.
ARMS = [
    ("subsets_poolB14.json", "uncertainty", BLUE),
    ("subsets_poolE_14.json", "expected activity", VERM),
    ("subsets_poolR_14.json", "random (control)", GREY),
]
RES_ORDER = ["motif_shared_core", "random", "evoaug", "genomic"]

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/analysis/nov25")
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--name", default="fig21_selection_subsets.png")
    a = ap.parse_args()
    src = REPO / a.src if not Path(a.src).is_absolute() else Path(a.src)
    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    loaded = []
    for fn, label, colour in ARMS:
        p = src / fn
        if not p.exists():
            print(f"missing {p}")
            continue
        loaded.append((json.loads(p.read_text()), label, colour))
    if not loaded:
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    # --- reservoir level ---------------------------------------------------
    ax = axes[0]
    names = [r for r in RES_ORDER if r in loaded[0][0].get("reservoir", {})]
    y = np.arange(len(names))
    h = 0.78 / max(1, len(loaded))
    for k, (d, label, colour) in enumerate(loaded):
        vals = [d["reservoir"][r]["ratio"] or 0 for r in names]
        ax.barh(y + k * h - 0.39 + h / 2, vals, height=h * 0.86, color=colour, label=label)
    ax.axvline(1.0, color="black", lw=1.1, zorder=3)
    ax.text(1.02, -0.72, "pool share", fontsize=7.5, color=GREY, va="center")
    ax.set_yticks(y, [n.replace("_", " ") for n in names])
    ax.set_ylim(len(names) - 0.42, -0.95)
    ax.set_xlabel("x the share the pool offered")
    ax.set_title("which reservoir the rule takes from\n(equal 4-way pool, 75,000 each)")

    # --- within motif_shared_core -----------------------------------------
    ax = axes[1]
    for d, label, colour in loaded:
        w = d.get("within", {}).get("motif_shared_core", {})
        f = w.get("factors", {}).get("n_motifs_planted", {})
        # Levels the sampler never produced carry pool_share 0 and a ratio of 0. Drawing
        # them would show three bars at zero and read as "the rule rejects sparse
        # plantings", when in fact no such sequence was ever on offer.
        lv = sorted((int(k), v) for k, v in f.items() if v["pool_share"] > 0)
        if not lv:
            continue
        ax.plot(
            [k for k, _ in lv],
            [v["ratio"] for _, v in lv],
            "o-",
            color=colour,
            lw=2.2,
            ms=5,
            label=f"{label}  (n={w.get('n_traced', 0):,})",
        )
    ax.axhline(1.0, color="black", lw=1.1, zorder=3)
    # A site count is an integer; half-ticks invite a reading between levels that the
    # sampler never produces.
    ticks = sorted(
        {
            int(k)
            for d, _, _ in loaded
            for k, v in d.get("within", {})
            .get("motif_shared_core", {})
            .get("factors", {})
            .get("n_motifs_planted", {})
            .items()
            if v["pool_share"] > 0
        }
    )
    if ticks:
        ax.set_xticks(ticks)
    ax.set_xlabel("motif sites planted in the sequence")
    ax.set_ylabel("x the share the pool offered")
    ax.set_title("and which sequences inside motif_shared_core")
    ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "An acquisition rule picks a reservoir, and then picks within it",
        y=1.0,
        x=0.02,
        ha="left",
        fontsize=12,
    )
    # Provenance, because this panel is only trustworthy where regeneration verified.
    fig.text(
        0.02,
        -0.04,
        "Right panel: motif_shared_core regenerated at its stored seed and checked "
        "byte-identical to the pool at 2,000 spread positions before any attribution. "
        "evoaug and genomic draw from a base pool whose order the stamp does not fix, so "
        "they do not reproduce and are deliberately not shown here.",
        fontsize=7.2,
        color=GREY,
        wrap=True,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncols=3,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out / a.name)
    plt.close(fig)
    print(f"{a.name} written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
