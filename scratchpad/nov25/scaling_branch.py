"""From-scratch curves with the acquisition branches drawn off the 30k pilot.

The two experiments answer different halves of one question and have never been on one
axis. The from-scratch curve says what a corpus is worth when you draw it; the acquisition
arms say what SELECTING the next increment is worth once you already have a 30k pilot. Drawn
together, a branch that sits below the from-scratch line at the same total size means
selection beat drawing, and that is the comparison the registered report has to make.

RECIPE HONESTY. Only cells at the tuned per-size recipe are drawn solid. The 60k
acquisition arms in menu_retrain_v0 are cfg e500_wd15, not the tuned e500 for that size, so
a point there would mix a recipe difference into the branch; they are drawn as open markers
and excluded from the fitted line. The 105k and 200k arms are e500, which is the tuned
recipe at 100k, so those carry the branch.

Also writes the compact heatmap: headline endpoints only, a few sizes, every strategy.

    python scratchpad/nov25/scaling_branch.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "orchestration"))
sys.path.insert(0, str(REPO / "scratchpad/nov25"))
from scaling_mse import RESERVOIRS, TUNED, cell_dir, endpoint_mse  # noqa: E402

MR = REPO / "outputs/menu_retrain_v0"
HEADLINE = [
    ("genomic", "genomic"),
    ("ood", "designed"),
    ("random_32k", "random"),
    ("D:inversion", "Δinversion"),
]
# Acquisition arm families: tag prefix -> (label, pilot reservoir, colour, style)
BRANCHES = {
    "it3_epi": ("uncertainty, 3 rounds", "genomic", "#0072B2", "-"),
    "os_epi": ("uncertainty, 1 round", "genomic", "#56B4E9", "-"),
    "os_random": ("random, 1 round", "genomic", "#999999", "-"),
    "os_badge": ("badge, 1 round", "genomic", "#CC79A7", "-"),
    "a54_it3_epi": ("uncertainty, 3 rounds (motif pilot)", "motif_shared_core", "#D55E00", "-"),
    "a54_os_random": ("random, 1 round (motif pilot)", "motif_shared_core", "#E69F00", "-"),
}
# Recipes that ARE the tuned choice at that size; anything else is drawn open.
TUNED_AT = {60_000: "e500", 105_000: "e500", 200_000: "e500"}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

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

CELL = re.compile(r"^bench38_ln_(?P<cfg>.+?)_(?P<size>\d+)k_c20_(?P<tag>.+?)(?:_s\d+)?$")


def branch_points(max_members: int) -> dict[str, dict[int, list[dict[str, float]]]]:
    """tag -> size -> list of per-replicate MSE dicts."""
    out: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for t in sorted(MR.glob("*/acq_trace.json")):
        d = t.parent
        m = CELL.match(d.name)
        if not m:
            continue
        tag, size, cfg = m.group("tag"), int(m.group("size")) * 1000, m.group("cfg")
        if tag not in BRANCHES or TUNED_AT.get(size) != cfg:
            continue
        if not glob.glob(str(d / "*_meta.json")):
            continue
        r = endpoint_mse(d, max_members)
        if r:
            out[tag][size].append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--max-members", type=int, default=3)
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    scratch: dict[str, dict[int, dict[str, float]]] = {}
    for res, suffix in RESERVOIRS:
        for D in sorted(TUNED):
            d = cell_dir(suffix, D)
            if d is None:
                continue
            r = endpoint_mse(d, a.max_members)
            if r:
                scratch.setdefault(res, {})[D] = r
    br = branch_points(a.max_members)
    print(f"from-scratch: {len(scratch)} reservoirs | branches: {sorted(br)}")

    # ---------------- fig15: curves with branches -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0))
    for ax, (ep, lab) in zip(axes.ravel(), HEADLINE, strict=False):
        for (res, _), colour in zip(RESERVOIRS, PALETTE, strict=False):
            pts = [
                (D, v[ep]) for D, v in sorted(scratch.get(res, {}).items()) if ep in v and v[ep] > 0
            ]
            if len(pts) >= 2:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "o-",
                    color=colour,
                    ms=4,
                    lw=1.4,
                    alpha=0.55,
                    label=f"from scratch: {res}",
                )
        for tag, (blab, pilot, colour, style) in BRANCHES.items():
            sizes = sorted(br.get(tag, {}))
            ys = [float(np.mean([r[ep] for r in br[tag][s] if ep in r])) for s in sizes]
            keep = [(s, y) for s, y in zip(sizes, ys, strict=True) if np.isfinite(y) and y > 0]
            if not keep:
                continue
            anchor = scratch.get(pilot, {}).get(30_000, {}).get(ep)
            xs = [30_000] + [k[0] for k in keep] if anchor else [k[0] for k in keep]
            yv = [anchor] + [k[1] for k in keep] if anchor else [k[1] for k in keep]
            ax.plot(xs, yv, style, color=colour, lw=2.2, marker="D", ms=5, label=blab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lab)
        ax.grid(True, which="both", alpha=0.18, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("total training sequences")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (stacked ensemble)")
    h, lbl = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lbl, loc="lower center", ncols=3, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(
        "From-scratch curves, with acquisition branching off the 30k pilot",
        y=0.98,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig15_scaling_with_branches.png")
    plt.close(fig)
    print("fig15 written")

    # ---------------- fig16: compact headline heatmap ----------------------------
    sizes = [10_000, 30_000, 100_000, 300_000]
    cols = [(res, D) for res, _ in RESERVOIRS for D in sizes if D in scratch.get(res, {})]
    M = np.full((len(HEADLINE), len(cols)), np.nan)
    for j, (res, D) in enumerate(cols):
        for i, (ep, _) in enumerate(HEADLINE):
            M[i, j] = scratch[res][D].get(ep, np.nan)
    # Normalised per endpoint ROW by the best cell in the row, so one colour scale works
    # across endpoints whose MSE differs by two orders of magnitude.
    R = M / np.nanmin(M, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.0 + 0.42 * len(cols), 3.3))
    im = ax.imshow(R, aspect="auto", cmap="RdYlGn_r", vmin=1.0, vmax=np.nanpercentile(R, 92))
    ax.set_yticks(range(len(HEADLINE)), [h[1] for h in HEADLINE])
    ax.set_xticks(
        range(len(cols)),
        [f"{res}\n{D // 1000}k" for res, D in cols],
        rotation=90,
        fontsize=6.5,
    )
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isfinite(R[i, j]):
                ax.text(
                    j,
                    i,
                    f"{R[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if R[i, j] > np.nanpercentile(R, 75) else "black",
                )
    ax.set_title("Headline endpoints: MSE relative to the best cell in each row", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="x best")
    fig.savefig(out / "fig16_headline_heatmap.png")
    plt.close(fig)
    print("fig16 written")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
