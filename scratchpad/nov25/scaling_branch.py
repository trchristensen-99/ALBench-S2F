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
import json
import os
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
    ("ood", "high-activity designed"),
    ("random_32k", "random"),
    ("D:inversion", "Δinversion"),
]
# Acquisition arm families: tag -> (label, colour). The PILOT is NOT hardcoded -- it is read
# from each trace's own `seed_pool`, because the tags do not say what they branch from and
# guessing was wrong: the bare `it3_epi`/`os_epi`/`os_random` arms at 105k seed from
# motif_shared_core_d30000, not genomic, while `os_badge`/`os_kmeanspp` seed from
# genomic_d30000, and every 200k arm seeds from motif_shared_core_d100000 rather than from
# 30k at all. Anchoring a branch at the wrong from-scratch point makes selection look better
# or worse than it is.
BRANCHES = {
    "it3_epi": ("uncertainty, 3 rounds", "#0072B2"),
    "os_epi": ("uncertainty, 1 round", "#56B4E9"),
    "os_random": ("random, 1 round", "#999999"),
    "os_badge": ("badge, 1 round", "#CC79A7"),
    "os_kmeanspp": ("diversity (k-means++), 1 round", "#009E73"),
    "a54_it3_epi": ("uncertainty, 3 rounds (a54)", "#D55E00"),
    "a54_os_random": ("random, 1 round (a54)", "#E69F00"),
}
# Which from-scratch curve a seed_pool corresponds to, and at what size.
SEED_POOL = {
    "motif_shared_core_d30000.npz": ("motif_shared_core", 30_000),
    "genomic_d30000.npz": ("genomic", 30_000),
    "motif_shared_core_d100000.npz": ("motif_shared_core", 100_000),
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


def branch_points(max_members: int):
    """(tag, seed_pool) -> size -> list of per-replicate MSE dicts.

    Keyed by seed_pool as well as tag, because one tag spans several pilots across sizes and
    averaging those together would silently merge two different experiments.
    """
    out: dict[tuple[str, str], dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
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
        try:
            tr = json.loads(t.read_text())
            sp = os.path.basename(tr.get("seed_pool", ""))
        except Exception:
            sp = ""
        if sp not in SEED_POOL:
            continue
        r = endpoint_mse(d, max_members)
        if r:
            out[(tag, sp)][size].append(r)
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
        for (tag, sp), per_size in sorted(br.items()):
            blab, colour = BRANCHES[tag]
            pilot_res, pilot_D = SEED_POOL[sp]
            sizes = sorted(per_size)
            ys = [float(np.mean([r[ep] for r in per_size[s] if ep in r])) for s in sizes]
            keep = [(s, y) for s, y in zip(sizes, ys, strict=True) if np.isfinite(y) and y > 0]
            if not keep:
                continue
            anchor = scratch.get(pilot_res, {}).get(pilot_D, {}).get(ep)
            xs = ([pilot_D] if anchor else []) + [k[0] for k in keep]
            yv = ([anchor] if anchor else []) + [k[1] for k in keep]
            ls = "-" if pilot_res == "motif_shared_core" else "--"
            ax.plot(
                xs,
                yv,
                ls,
                color=colour,
                lw=2.2,
                marker="D",
                ms=5,
                label=f"{blab} | pilot {pilot_res} {pilot_D // 1000}k",
            )
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

    # ---------------- fig16: one panel per dataset size --------------------------
    # Split by size, because a single wide strip invited comparisons across sizes that are
    # not the comparison anyone wants: within a size the strategies are directly comparable,
    # across sizes the data amount differs and the colour says nothing useful. Each panel is
    # normalised by the best STRATEGY at that size for that endpoint, so 1.00 always means
    # "the best corpus at this size" and the number is a straightforward MSE ratio.
    sizes = [s for s in (10_000, 30_000, 100_000, 300_000) if any(s in v for v in scratch.values())]
    res_present = [r for r, _ in RESERVOIRS if r in scratch]
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(3.05 * len(sizes), 3.4), squeeze=False, sharey=True
    )
    for ax, D in zip(axes[0], sizes, strict=True):
        M = np.full((len(HEADLINE), len(res_present)), np.nan)
        for j, r in enumerate(res_present):
            for i, (ep, _) in enumerate(HEADLINE):
                M[i, j] = scratch[r].get(D, {}).get(ep, np.nan)
        R = M / np.nanmin(M, axis=1, keepdims=True)
        im = ax.imshow(R, aspect="auto", cmap="RdYlGn_r", vmin=1.0, vmax=1.6)
        ax.set_xticks(range(len(res_present)), res_present, rotation=90, fontsize=7)
        ax.set_title(f"D = {D // 1000}k", fontsize=10)
        ax.grid(False)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if np.isfinite(R[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{R[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if R[i, j] > 1.4 else "black",
                    )
    axes[0][0].set_yticks(range(len(HEADLINE)), [h[1] for h in HEADLINE], fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02, label="MSE / best at this size")
    fig.suptitle(
        "Headline endpoints: MSE relative to the best corpus AT EACH SIZE",
        y=1.02,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.savefig(out / "fig16_headline_heatmap.png")
    plt.close(fig)
    print("fig16 written")

    # ---------------- fig15b: one pilot, few lines -------------------------------
    # The uncluttered version. Restricted to arms sharing ONE pilot
    # (motif_shared_core 30k) so every line starts from the same point and the only
    # difference is the rule -- which is the comparison, and which the full figure buries
    # under six from-scratch curves and three other pilots. The random arm is the control:
    # it draws from the same 1M candidate pool, so uncertainty minus random is the value of
    # selecting rather than the value of the candidate pool.
    PILOT = "motif_shared_core_d30000.npz"
    WANT = ["os_random", "os_epi", "it3_epi"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    for ax, (ep, lab) in zip(axes.ravel(), HEADLINE, strict=False):
        pts = [
            (D, v[ep])
            for D, v in sorted(scratch.get("motif_shared_core", {}).items())
            if ep in v and v[ep] > 0
        ]
        if len(pts) >= 2:
            ax.plot(
                [q[0] for q in pts],
                [q[1] for q in pts],
                "o-",
                color="#999999",
                ms=5,
                lw=1.8,
                label="from scratch: motif_shared_core",
            )
        for tag in WANT:
            per_size = br.get((tag, PILOT))
            if not per_size:
                continue
            blab, colour = BRANCHES[tag]
            sizes_b = sorted(per_size)
            ys = [float(np.mean([r[ep] for r in per_size[s] if ep in r])) for s in sizes_b]
            keep = [(s, y) for s, y in zip(sizes_b, ys, strict=True) if np.isfinite(y) and y > 0]
            if not keep:
                continue
            anchor = scratch.get("motif_shared_core", {}).get(30_000, {}).get(ep)
            xs = ([30_000] if anchor else []) + [q[0] for q in keep]
            yv = ([anchor] if anchor else []) + [q[1] for q in keep]
            ax.plot(xs, yv, "-", color=colour, lw=2.4, marker="D", ms=6, label=blab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lab)
        ax.grid(True, which="both", alpha=0.18, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("total training sequences")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (stacked ensemble)")
    h, lb = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncols=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Selection vs drawing, from one 30k motif_shared_core pilot",
        y=0.98,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig15b_selection_vs_drawing.png")
    plt.close(fig)
    print("fig15b written")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
