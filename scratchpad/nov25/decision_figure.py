"""fig24: what we would actually order, and how much of it is extrapolation.

The deliverable is a training corpus, and a corpus is a number of oligos per reservoir.
Everything measured so far tops out at 300,000 rows; the order is 3.2 million. So this
figure has to do three things and keep them visibly separate:

  A  the measured curves, extended to the order size, with the extrapolation drawn
     differently from the data
  B  how far the extrapolation can be trusted, measured rather than asserted -- refit each
     curve on 10k/30k/100k only, predict the 300k point we actually hold, and read off the
     error. That is one decade of extrapolation checked against ground truth; the order is
     another decade beyond it, so the band is a floor on the uncertainty, not a bound.
  C  a draft order, built from the endpoint-relative winners rather than from a single
     ranking -- no reservoir wins every endpoint (fig19), so the order is a portfolio and
     the weights are the decision the team still has to make.

THE CAPACITY WALL IS NOT A MODELLING CHOICE. There are ~315,000 distinct real CREs. A
genomic curve cannot be extended past that at any price, so the panel draws a wall there
and the allocator will not spend past it. This is the single hardest constraint on the
order and it is the reason the question is "which synthetic reservoirs" rather than
"how much real sequence".

    python scratchpad/nov25/decision_figure.py --out outputs/figures/nov25 --budget 3200000
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
BLUE, VERM, GREEN, PURPLE, ORANGE, SKY = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
GREY = "#6b6b6b"

COLOUR = {
    "motif_shared_core": BLUE,
    "motif_ct_enriched": SKY,
    "evoaug_paper2023": VERM,
    "random": GREEN,
    "genomic": PURPLE,
    "mutagenesis": ORANGE,
}
# Distinct sequences each reservoir can supply. genomic is the measured count of real CREs
# in the Gosai library; the synthetic reservoirs reseed, and 0 means "no ceiling we have
# hit". zoonomia_orthologs is absent from this figure entirely: its pools are 15.4% exact
# duplicates at every size, so its nominal capacity is not a usable one.
CAPACITY = {"genomic": 315_000}

# The endpoints a library has to serve, and what each one is FOR. The weights are the open
# decision; these are a starting point, not a recommendation.
OBJECTIVES = {
    "balanced": {"genomic": 1.0, "ood": 1.0, "random_32k": 1.0, "D:evoaug_mix": 1.0},
    "designed-led": {"ood": 3.0, "genomic": 1.0, "random_32k": 0.5, "D:evoaug_mix": 1.0},
    "variant-led": {"D:evoaug_mix": 3.0, "genomic": 1.5, "ood": 0.5, "random_32k": 0.5},
    # Not a weighting but a constraint, and the one the team will actually argue about.
    # The four endpoints above are all model-generalisation endpoints, and real CREs lose
    # every one of them -- so an allocator driven only by those orders zero real sequence.
    # Reserving the whole real-CRE pool first prices that decision instead of hiding it.
    "balanced,\nall real CREs": {
        "genomic": 1.0,
        "ood": 1.0,
        "random_32k": 1.0,
        "D:evoaug_mix": 1.0,
    },
}
# Objectives that pre-commit part of the budget before anything is optimised.
FORCED = {"balanced,\nall real CREs": {"genomic": 315_000}}
EP_LABEL = {
    "genomic": "genomic (held-out chr)",
    "ood": "high-activity designed",
    "random_32k": "random",
    "D:evoaug_mix": "EvoAug-style Δ",
}

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


def fit(sizes: list[int], vals: list[float]) -> tuple[float, float]:
    """log MSE = a + b log D. Two parameters on four points; b is the scaling exponent."""
    x, y = np.log10(sizes), np.log10(vals)
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def predict(a: float, b: float, D: float) -> float:
    return float(10 ** (a + b * np.log10(D)))


def holdout_error(data: dict, endpoints: list[str]) -> dict[str, float]:
    """Refit on 10k/30k/100k, predict 300k, return predicted/actual per reservoir+endpoint.

    This is the only honest statement available about the extrapolation: one decade out of
    sample, against a point we hold. Reported as a ratio so it composes with the predicted
    value directly.
    """
    out: dict[str, float] = {}
    for res, per in data.items():
        for ep in endpoints:
            pts = [(int(k), per[k][ep]) for k in per if ep in per[k] and per[k][ep] > 0]
            pts.sort()
            train = [(d, v) for d, v in pts if d < 300_000]
            got = dict(pts).get(300_000)
            if len(train) < 3 or not got:
                continue
            a, b = fit([d for d, _ in train], [v for _, v in train])
            out[f"{res}|{ep}"] = predict(a, b, 300_000) / got
    return out


def allocate(
    data: dict, weights: dict[str, float], budget: int, forced: dict[str, int] | None = None
) -> dict[str, int]:
    """Spend the budget on each endpoint's own best reservoir, capped by capacity.

    Deliberately NOT a joint optimisation over mixtures. Every mixture measured so far
    (35 arms, three rules, fig5b) failed to beat the best single reservoir, so an optimiser
    that assumed mixing adds value would be inventing its headline. This allocates by the
    one thing that IS measured: which reservoir wins which endpoint.
    """
    alloc: dict[str, int] = dict(forced or {})
    budget = max(0, budget - sum(alloc.values()))
    tot = sum(weights.values()) or 1
    for ep, w in weights.items():
        share = int(budget * w / tot)
        # Rank reservoirs by predicted MSE at the size this endpoint's share would buy.
        ranked = []
        for res, per in data.items():
            pts = sorted((int(k), per[k][ep]) for k in per if ep in per[k] and per[k][ep] > 0)
            if len(pts) < 3:
                continue
            a, b = fit([d for d, _ in pts], [v for _, v in pts])
            cap = CAPACITY.get(res, 10**9)
            ranked.append((predict(a, b, min(share, cap)), res, cap))
        ranked.sort()
        if not ranked:
            # An endpoint nothing can be scored on silently swallowed its whole share once:
            # a stale scaling_mse.json had no D:evoaug_mix column, so a quarter of a 3.2M
            # order went unallocated and the only visible symptom was a total that did not
            # add up. Refuse rather than return a short order that looks complete.
            raise SystemExit(
                f"no reservoir has a usable curve for endpoint {ep!r}, so its "
                f"{share:,}-oligo share cannot be allocated; regenerate scaling_mse.json "
                "with the sidecar endpoints before trusting this figure"
            )
        left = share
        for _, res, cap in ranked:
            if left <= 0:
                break
            room = cap - alloc.get(res, 0)
            take = int(min(left, room))
            if take <= 0:
                continue
            alloc[res] = alloc.get(res, 0) + take
            left -= take
        if left > 0:
            raise SystemExit(
                f"endpoint {ep!r} could not spend {left:,} of its {share:,} oligos: every "
                "reservoir it ranks hit its capacity. The order would be short, so this is "
                "raised rather than drawn as a complete bar."
            )
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/figures/nov25/scaling_mse.json")
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--budget", type=int, default=3_200_000)
    ap.add_argument("--endpoint", default="ood", help="endpoint drawn in panel A")
    ap.add_argument("--name", default="fig24_what_we_would_order.png")
    a = ap.parse_args()
    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((REPO / a.src).read_text())
    eps = list(EP_LABEL)

    fig = plt.figure(figsize=(14.2, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.25], wspace=0.30)

    # ---- A: measured, then extrapolated ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ep = a.endpoint
    for res, per in data.items():
        pts = sorted((int(k), per[k][ep]) for k in per if ep in per[k] and per[k][ep] > 0)
        if len(pts) < 3:
            continue
        c = COLOUR.get(res, GREY)
        ax.plot(
            [d for d, _ in pts],
            [v for _, v in pts],
            "o-",
            color=c,
            lw=2.0,
            ms=5,
            label=res.replace("_", " "),
        )
        fa, fb = fit([d for d, _ in pts], [v for _, v in pts])
        cap = CAPACITY.get(res, a.budget)
        xs = np.logspace(np.log10(pts[-1][0]), np.log10(min(cap, a.budget)), 40)
        ax.plot(xs, [predict(fa, fb, x) for x in xs], ":", color=c, lw=1.8)
        if res in CAPACITY:
            ax.axvline(cap, color=c, lw=1.2, ls="-", alpha=0.5)
            # The share matters more than the count: even reserving every real CRE in
            # existence, they are under a tenth of the order, so the other 90% has to be
            # synthetic whatever anyone prefers.
            ax.text(
                cap * 0.93,
                ax.get_ylim()[1],
                f"  {res} runs out at {cap // 1000}k\n  = {cap / a.budget:.1%} of the order",
                fontsize=7,
                color=c,
                ha="right",
                va="top",
            )
    ax.axvline(a.budget, color="black", lw=1.4)
    ax.text(
        a.budget * 0.95,
        0.04,
        f"order size\n{a.budget / 1e6:.1f}M",
        fontsize=7.5,
        ha="right",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training sequences")
    ax.set_ylabel("MSE (stacked ensemble)")
    ax.set_title(f"A. measured to 300k, extrapolated past it\n{EP_LABEL.get(ep, ep)}", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")

    # ---- B: how wrong the extrapolation already is ----------------------
    ax = fig.add_subplot(gs[0, 1])
    err = holdout_error(data, eps)
    if err:
        vals = np.array(list(err.values()))
        ax.axhline(1.0, color="black", lw=1.2)
        xs = np.arange(len(vals))
        order = np.argsort(vals)
        cols = [COLOUR.get(list(err)[i].split("|")[0], GREY) for i in order]
        ax.bar(xs, vals[order], color=cols, width=0.82)
        ax.set_xticks([])
        ax.set_ylabel("predicted / actual at 300k")
        lo, hi = float(vals.min()), float(vals.max())
        ax.set_title(
            "B. one decade out of sample,\nchecked against the 300k point we hold", fontsize=10
        )
        # Headroom above the tallest bar, and the note placed in it -- at the top of the
        # axes it sat across the 1.0 reference line and the bars themselves.
        ax.set_ylim(0, max(1.05, hi) * 1.42)
        ax.text(
            0.03,
            0.985,
            f"fit on 10k/30k/100k only, {len(vals)} reservoir x endpoint curves\n"
            f"spans {lo:.2f}x to {hi:.2f}x: the law is OPTIMISTIC out of sample\n"
            f"and the order is another decade further out again",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            color=GREY,
        )

    # ---- C: the draft order ---------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    names = list(OBJECTIVES)
    allocs = [allocate(data, OBJECTIVES[n], a.budget, FORCED.get(n)) for n in names]
    reservoirs = sorted(
        {r for al in allocs for r in al}, key=lambda r: -max(al.get(r, 0) for al in allocs)
    )
    bottom = np.zeros(len(names))
    for res in reservoirs:
        vals = np.array([al.get(res, 0) / 1e6 for al in allocs])
        ax.barh(
            np.arange(len(names)),
            vals,
            left=bottom,
            height=0.62,
            color=COLOUR.get(res, GREY),
            label=res.replace("_", " "),
        )
        for i, v in enumerate(vals):
            if v > a.budget / 1e6 * 0.10:
                ax.text(
                    bottom[i] + v / 2,
                    i,
                    f"{v:.2f}M",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                    fontweight="bold",
                )
            elif v > 0:
                # A thin segment's label does not fit inside it, and centring it anyway
                # pushed the text past the bar's end where it read as the next row's value.
                ax.text(
                    bottom[i] + v + a.budget / 1e6 * 0.015,
                    i,
                    f"{v:.2f}M",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color=COLOUR.get(res, GREY),
                )
        bottom += vals
    ax.set_yticks(np.arange(len(names)), [n.replace("-", "\n") for n in names])
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xlim(0, a.budget / 1e6 * 1.16)
    free = [al for n, al in zip(names, allocs, strict=True) if n not in FORCED]
    if free and all(al.get("genomic", 0) == 0 for al in free):
        # The most provocative thing this panel says, said out loud rather than left to be
        # noticed from a missing colour.
        ax.text(
            0.5,
            1.03,
            "real CREs appear only in the row that reserves them: they lose all "
            "four\nmodelling endpoints, which is not the same as being useless",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=GREY,
        )
    ax.set_xlabel(f"oligos (millions of {a.budget / 1e6:.1f}M)")
    ax.set_title(
        f"C. a draft order under {len(names)} objectives\n(the weights are the open decision)",
        fontsize=10,
        pad=26,
    )
    # Below the axes: inside, it sat on top of the bottom bar.
    ax.legend(fontsize=7.5, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    fig.suptitle(
        "What a 3.2M order would contain, and which parts of that are measured",
        y=1.02,
        x=0.02,
        ha="left",
        fontsize=12.5,
    )
    fig.text(
        0.02,
        -0.07,
        "Panel A's dotted segments are extrapolation, not data. Panel C allocates each "
        "objective's weight to that endpoint's best-predicted reservoir, capped by how many "
        "distinct sequences the reservoir can supply; it does not assume mixing adds value, "
        "because no mixture measured so far has beaten the best single reservoir (fig5b, 35 "
        "arms). genomic stops at 315,000 because that is how many real CREs exist in the "
        "library.",
        fontsize=7.4,
        color=GREY,
        wrap=True,
    )
    fig.savefig(out / a.name, bbox_inches="tight")
    plt.close(fig)
    print(f"{a.name} written -> {out}")
    for n, al in zip(names, allocs, strict=True):
        tot = sum(al.values())
        print(
            f"  {n:14s} "
            + "  ".join(f"{r}={v / 1e6:.2f}M" for r, v in sorted(al.items(), key=lambda kv: -kv[1]))
            + f"   (total {tot / 1e6:.2f}M)"
        )
    if err:
        v = np.array(list(err.values()))
        print(
            f"  extrapolation check: predicted/actual at 300k spans "
            f"{v.min():.2f}x-{v.max():.2f}x over {len(v)} curves"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
