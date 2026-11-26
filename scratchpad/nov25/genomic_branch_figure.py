"""fig22: acquisition from one starting corpus, named by what it was allowed to choose from.

Every earlier version of this figure mixed two starting corpora and labelled the curves by
rule alone, so "uncertainty, 1 round" appeared twice meaning two different experiments, and
the thing that actually varied -- the reservoir set on offer -- was not in the caption at
all. Here the start is fixed at 30,000 real CREs and the legend names the pool:

    genomic only            270,000 more real CREs; the reservoir's whole remaining capacity
    even 7-way union        100,000 each from seven reservoirs
    union, uncertainty-weighted
                            the same seven, in the proportions uncertainty chose for itself
                            when it was handed the even union (fig18)

BADGE is not drawn. Its own controls settled it: badge scores below badge_epistemic_only,
so the diversity term subtracts rather than adds, and showing a fourth line that is already
known to be dominated spends attention that the pool axis needs.

Two modes, because the labels live on the cluster and the figure is iterated locally:

    # on the cluster
    python scratchpad/nov25/genomic_branch_figure.py --emit summary.json
    # anywhere
    python scratchpad/nov25/genomic_branch_figure.py --from summary.json --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KOO = Path(os.environ.get("ALBENCH_KOO", "/grid/koo/home/christen/nov25_extra"))

BLUE, VERM, GREEN, PURPLE, ORANGE, SKY = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
GREY = "#6b6b6b"

# pool tag -> (legend text, colour). The text says the RESERVOIR SET, because that is the
# axis the figure exists to separate.
POOLS = {
    "g30_own": ("genomic only (270k more real CREs)", BLUE),
    "g30_u7": ("even 7-way union (100k each)", VERM),
    "g30_umix": ("union, weighted to uncertainty's own mixture", GREEN),
}
RULES = {
    "random": ("random", ":"),
    "epistemic": ("uncertainty", "-"),
    "expected_activity": ("expected activity", "--"),
}
HEADLINE = [
    ("genomic", "genomic (held-out chromosomes)"),
    ("ood", "high-activity designed"),
    ("random_32k", "random"),
    ("D:evoaug_mix", "EvoAug-style Δ (real effects)"),
]


def emit(path: str, max_members: int) -> int:
    """Score every genomic-start arm and write one JSON the figure can be built from."""
    sys.path.insert(0, str(REPO / "scratchpad/nov25"))
    from scaling_mse import endpoint_mse  # noqa: E402

    rec: dict = {"arms": {}, "scratch": {}, "pools": {}}
    for pool_tag in POOLS:
        mix = KOO / "pools" / f"{pool_tag}_mix.json"
        if mix.exists():
            rec["pools"][pool_tag] = json.loads(mix.read_text())
    for tag_dir in sorted((KOO / "outputs/ac60").glob("*")):
        tag = tag_dir.name
        inner = [Path(p) for p in glob.glob(str(tag_dir / "bench38_*")) if Path(p).is_dir()]
        if not inner:
            continue
        d = inner[0]
        if not glob.glob(str(d / "*_meta.json")):
            continue
        r = endpoint_mse(d, max_members)
        if not r:
            continue
        rec["arms"][tag] = {"mse": r, "cell": d.name}
        print(f"  {tag:34s} {len(r)} endpoints")
        # The realised mixture, where the candidate pool recorded one. run_al now carries
        # the `source` column onto its selection; for arms selected before that change it
        # is recovered by matching sequences back to the pool they came from.
        sel = KOO / "pools" / f"sel_{tag}_d60000.npz"
        pool_tag = "_".join(tag.split("_")[:2])
        cand = KOO / "pools" / f"{pool_tag}.npz"
        if sel.exists() and cand.exists():
            got = _realised(sel, cand)
            if got:
                rec["arms"][tag]["realised"] = got
                print(f"      realised: {got}")

    # The two from-scratch anchors: 30k (where every arm starts) and 60k (what the same
    # budget buys with no selection at all).
    from scaling_mse import cell_dir  # noqa: E402

    d30 = cell_dir("", 30_000)
    if d30 is not None:
        rec["scratch"]["30000"] = endpoint_mse(d30, max_members)
    g60 = KOO / "outputs/ac60/g60_scratch"
    inner = [Path(p) for p in glob.glob(str(g60 / "bench38_*")) if Path(p).is_dir()]
    if inner and glob.glob(str(inner[0] / "*_meta.json")):
        rec["scratch"]["60000"] = endpoint_mse(inner[0], max_members)
        print("  g60_scratch scored")
    Path(path).write_text(json.dumps(rec, indent=1))
    print(f"-> {path}")
    return 0


def _realised(sel: Path, cand: Path) -> dict | None:
    import numpy as np

    zs = np.load(sel, allow_pickle=True)
    if "source" in zs.files:
        s = np.asarray(zs["source"])[30_000:]
    else:
        zc = np.load(cand, allow_pickle=True)
        if "source" not in zc.files:
            return None
        cs, cq = np.asarray(zc["source"]), np.asarray(zc["sequences"])
        where = {q: i for i, q in enumerate(map(str, cq))}
        idx = [where[q] for q in map(str, np.asarray(zs["sequences"])[30_000:]) if q in where]
        if len(idx) < 1000:
            return None
        s = cs[np.array(idx)]
    n = len(s)
    out: dict[str, float] = {}
    for r in set(map(str, s)):
        out[r] = round(float((s == r).sum()) / n, 4)
    return out


def draw(src: str, out_dir: Path, name: str) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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
    rec = json.loads(Path(src).read_text())
    arms, scratch = rec["arms"], rec["scratch"]
    if not arms:
        print("no scored arms yet")
        return 1

    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(4.0 * len(HEADLINE), 4.4), sharex=True)
    for ax, (ep, lab) in zip(np.atleast_1d(axes), HEADLINE, strict=False):
        a30 = scratch.get("30000", {}).get(ep)
        a60 = scratch.get("60000", {}).get(ep)
        if a30 and a60:
            # The control line: same 60,000 rows, chosen by nobody. Drawn first and in
            # black so every coloured line is read against it rather than against each
            # other -- a rule that lands on this line selected nothing of value.
            ax.plot(
                [30_000, 60_000],
                [a30, a60],
                "-",
                color="black",
                lw=2.0,
                zorder=5,
                marker="s",
                ms=5,
                label="no selection (genomic's own next 30k)",
            )
        elif a30:
            ax.plot([30_000], [a30], "s", color="black", ms=6, zorder=5, label="30k start")
        for pool_tag, (plab, colour) in POOLS.items():
            for rule, (rlab, ls) in RULES.items():
                v = arms.get(f"{pool_tag}_{rule}", {}).get("mse", {}).get(ep)
                if not v or not a30:
                    continue
                ax.plot(
                    [30_000, 60_000],
                    [a30, v],
                    ls,
                    color=colour,
                    lw=2.0,
                    marker="D",
                    ms=4.5,
                    alpha=0.95,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([30_000, 60_000], ["30k", "60k"])
        ax.grid(True, which="both", alpha=0.16, lw=0.5)
    np.atleast_1d(axes)[0].set_ylabel("MSE (stacked ensemble, lower is better)")
    for ax in np.atleast_1d(axes):
        ax.set_xlabel("total training sequences")

    # Two legends, because colour and line style mean different things and one combined
    # legend of nine entries hides that.
    from matplotlib.lines import Line2D

    pool_h = [Line2D([], [], color=c, lw=2.4) for _, (_, c) in POOLS.items()]
    pool_l = [t for _, (t, _) in POOLS.items()]
    pool_h.insert(0, Line2D([], [], color="black", lw=2.0, marker="s", ms=5))
    pool_l.insert(0, "no selection: genomic's own next 30k")
    rule_h = [Line2D([], [], color=GREY, lw=2.0, ls=s) for _, (_, s) in RULES.items()]
    rule_l = [t for _, (t, _) in RULES.items()]
    leg1 = fig.legend(
        pool_h,
        pool_l,
        loc="lower left",
        bbox_to_anchor=(0.02, -0.10),
        fontsize=8.5,
        title="reservoir set on offer",
        title_fontsize=8.5,
        alignment="left",
    )
    leg1._legend_box.align = "left"
    fig.legend(
        rule_h,
        rule_l,
        loc="lower right",
        bbox_to_anchor=(0.98, -0.10),
        fontsize=8.5,
        title="selection rule",
        title_fontsize=8.5,
        alignment="left",
    )
    fig.suptitle(
        "Everything starts from the same 30,000 real CREs; only the pool and the rule differ",
        y=1.0,
        x=0.02,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out_dir / name)
    plt.close(fig)
    print(f"{name} written -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default="")
    ap.add_argument("--from", dest="src", default="outputs/analysis/nov25/genomic_branches.json")
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--name", default="fig22_genomic_start_acquisition.png")
    ap.add_argument("--max-members", type=int, default=3)
    a = ap.parse_args()
    if a.emit:
        return emit(a.emit, a.max_members)
    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = a.src if Path(a.src).is_absolute() else str(REPO / a.src)
    return draw(src, out, a.name)


if __name__ == "__main__":
    raise SystemExit(main())
