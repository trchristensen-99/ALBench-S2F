"""Build the figure set for the 2026-11-25 lab meeting.

Each panel answers one of the three questions raised at the last meeting, and each is
re-runnable: an arm that has not finished is skipped with a note rather than faked, so this
can be run again as the afternoon's jobs land.

    EVAL SET        fig1  how independent the endpoints actually are, measured across the
                          55 corpora we now have, and how much each one discriminates
    MULTI-CELL      fig6  the conditioning gate: channel vs content, on seen cells
                    fig7  leave-one-cell-out: does a description reach an unseen cell
                    fig8  the data-poor allocation curve
    SHORTEST PATH   fig2  within-family spread -- which member of each family to test
                    fig3  the density ladder, the one parameter that moved anything
                    fig4  the tier result, against how much inactive content was planted
                    fig5  acquisition: candidate-pool size, and the rules
                    fig9  design-method budget: where the guide and the oracle part ways

Colours are Okabe-Ito throughout (colourblind-safe by construction, and the standard in
this literature), assigned in a fixed order per figure and never cycled.

    python scratchpad/nov25/make_meeting_figures.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "orchestration"))
sys.path.insert(0, str(REPO / "scripts/analysis"))
sys.path.insert(0, str(REPO / "scratchpad/nov24"))
sys.path.insert(0, str(REPO / "scratchpad/nov25"))
from read_acq52 import SETS, read_run  # noqa: E402

MR = REPO / "outputs/menu_retrain_v0"
VAR = REPO / "outputs/var41"
PREFIX = "bench38_ln_e500_wd2_30k_c20"
PRETTY = {"genomic": "genomic", "ood": "designed", "random_32k": "random"}

# Okabe-Ito, light-surface safe (yellow omitted: it fails contrast on white).
BLUE, VERM, GREEN, PURPLE, ORANGE, SKY = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
GREY = "#6b6b6b"
SEED_NOISE = 0.008  # measured single-run sd on this harness, averaged over endpoints
# Per-endpoint seed noise: the mean within-arm sd over the three seeds, measured across the
# 36 candidate corpora by scratchpad/nov25/eval_panel_selection.py. These differ by a factor
# of seven, so the averaged constant above must not be used to compare endpoints.
NOISE = {"genomic": 0.0022, "ood": 0.0158, "random_32k": 0.0023}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


# --------------------------------------------------------------------------- data access

_CACHE: dict[Path, tuple | None] = {}


def stacked(d: Path) -> dict[str, float] | None:
    """Stacked r per eval set for one arm directory, or None if it has no members."""
    if d not in _CACHE:
        _CACHE[d] = read_run(d) if d.is_dir() else None
    got = _CACHE[d]
    return None if got is None else got[1]


def arm(name: str, root: Path = VAR) -> dict[str, float] | None:
    return stacked(root / f"{PREFIX}_{name}")


def note(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=GREY)
    ax.set_axis_off()


# --------------------------------------------------------------------------- fig 1

# Every corpus we can place on the same 30k axis: Arm A's families plus Arm A3's variants.
ARM_A = [
    "genomic",
    "random",
    "random_gc_sweep",
    "gc_matched",
    "dinuc_shuffle",
    "mutagenesis",
    "evoaug",
    "evoaug_ours",
    "evoaug_paper2023",
    "motif_shared_core",
    "motif_ct_enriched",
    "motif_syntax_core",
    "accessible_tiles_k562",
    "zoonomia_orthologs",
    "hg38_random",
    "evo2",
]


def collect_corpora() -> tuple[list[str], np.ndarray]:
    names, rows = [], []
    for n in ARM_A:
        s = arm(n, MR)
        if s:
            names.append(n)
            rows.append([s[k] for k in SETS])
    for p in sorted(VAR.glob(f"{PREFIX}_*")):
        s = stacked(p)
        if s:
            names.append(p.name[len(PREFIX) + 1 :])
            rows.append([s[k] for k in SETS])
    return names, np.array(rows)


def fig1(out: Path) -> None:
    names, M = collect_corpora()
    if len(M) < 10:
        print("fig1: too few corpora")
        return
    gi, di, ri = (SETS.index(s) for s in ("genomic", "ood", "random_32k"))
    lab = [PRETTY[s] for s in SETS]

    # A corpus that teaches the student almost nothing -- the pure-design arms sit at
    # genomic r 0.31-0.74 against the motif families' 0.89 -- pulls every endpoint down
    # together, and a correlation computed over that range says only "broken corpora are
    # broken everywhere". The decision is made among CANDIDATE corpora, so the redundancy
    # that matters is the one inside that range.
    ok = M[:, gi] >= 0.85
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), gridspec_kw={"wspace": 0.42})

    # (a) pairwise scatter, with the degenerate arms shown but marked.
    ax = axes[0]
    for m, colour, marker, name in (
        (ok, BLUE, "o", "designed"),
        (ok, VERM, "^", "random"),
    ):
        yidx = di if name == "designed" else ri
        ax.scatter(M[m, gi], M[m, yidx], s=20, c=colour, marker=marker, label=f"{name} (candidate)")
        ax.scatter(
            M[~m, gi],
            M[~m, yidx],
            s=18,
            facecolors="none",
            edgecolors=colour,
            marker=marker,
            alpha=0.55,
        )
    ax.axvline(0.85, color=GREY, lw=1, ls="--")
    ax.text(0.855, 0.02, "candidate corpora ->", fontsize=7, color=GREY)
    ax.set_xlabel("genomic held-out r")
    ax.set_ylabel("other endpoint r")
    ax.set_title(f"(a) {int(ok.sum())} candidates of {len(M)} corpora")
    ax.legend(loc="upper left", fontsize=7)

    # (b) correlation among the candidates only; the all-corpora value is printed under it
    # so the difference between the two readings is on the slide rather than hidden.
    ax = axes[1]
    C = np.corrcoef(M[ok].T)
    Call = np.corrcoef(M.T)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)), lab, rotation=30, ha="right")
    ax.set_yticks(range(len(lab)), lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(
                j,
                i,
                f"{C[i, j]:.2f}\n({Call[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(C[i, j]) > 0.6 else "black",
            )
    ax.set_title("(b) redundancy among candidates\n(all corpora in brackets)", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8, pad=0.04)

    # (c) discrimination: spread across CANDIDATE corpora divided by this endpoint's OWN
    # seed noise, measured as the within-arm sd over its three seeds. A single shared noise
    # constant is wrong here and not harmlessly so: the genomic endpoint's within-arm sd is
    # 0.0022 while the designed set's is 0.0158, a factor of seven, so a shared denominator
    # understates genomic and overstates designed. fig10 does this for all 31 endpoints.
    ax = axes[2]
    disc = [M[ok, SETS.index(s)].std() / max(NOISE[s], 1e-6) for s in SETS]
    ax.barh(lab, disc, color=[BLUE, VERM, GREEN], height=0.6)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    for i, v in enumerate(disc):
        ax.text(v + max(disc) * 0.02, i, f"{v:.1f}x", va="center", fontsize=8)
    ax.set_xlabel("candidate spread / that endpoint's seed noise")
    ax.set_title("(c) discrimination", fontsize=9)
    ax.set_xlim(0, max(disc) * 1.3)

    fig.suptitle(
        "Eval sets: measure everything, report the few that are hard, independent and "
        "discriminating",
        y=1.05,
        fontsize=10.5,
    )
    fig.savefig(out / "fig1_endpoint_independence.png")
    plt.close(fig)
    print("fig1 written")


# --------------------------------------------------------------------------- fig 2

FAMILIES: dict[str, tuple[str | None, list[str], str]] = {
    "motif: tiers": (
        "motif_ct_enriched",
        [
            "tier_balanced",
            "tier_active_only",
            "tier_nodecoy",
            "tier_heavy_inactive",
            "tier_with_other",
            "tier_fine_vocab",
            "tier_up_ct",
            "tier_up_shared",
            "tier_flat",
            "tseq_pure",
            "tseq_half",
            "tseq_mixed",
        ],
        BLUE,
    ),
    "motif: vocab + density": (
        "motif_ct_enriched",
        [
            "mot_core_ct20",
            "mot_core_ct50",
            "mot_ct_v30",
            "mot_ct_v120",
            "mot_ct_n1_2",
            "mot_ct_n8_15",
            "mot_ct_n15_25",
            "mot_ct_consensus",
            "mot_ct_mut05",
            "mot_ct_mut15",
            "mot_sc_n8_15",
            "mot_sc_n15_25",
            "mot_ct_n25_40",
            "mot_ct_n40_60",
        ],
        GREEN,
    ),
    "evoaug: perturbation": (
        "evoaug",
        [
            "ea_equal",
            "ea_point_only",
            "ea_inv_only",
            "ea_trans_only",
            "ea_tandem_only",
            "ea_indel_only",
            "ea_indel5",
            "ea_indel50",
        ],
        ORANGE,
    ),
    "mutagenesis: rate": ("mutagenesis", ["mut_r002", "mut_r20", "mut_ladder"], PURPLE),
    "model-based: design": (
        None,
        [
            "fsp_max",
            "fsp_target",
            "fsp_min",
            "fsp_rand_init",
            "sa_max",
            "sa_target",
            "ada_max",
            "ada_target",
        ],
        VERM,
    ),
    "model-based: selection": (
        None,
        [
            "ise_maximize_g30k",
            "ise_maximize_g300k",
            "ise_minimize",
            "ise_target_mid",
            "ise_spread",
            "ise_spread_g300k",
            "ise_explore",
        ],
        SKY,
    ),
}


def fig2(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    # Both motif reservoirs as reference lines, not just each family's own default: the best
    # member of the vocab/density family (`mot_core_ct50`) is a motif_shared_core variant,
    # so without shared_core's line a reader credits it with a gain that is just the other
    # reservoir.
    refs = []
    for res, ls in (("motif_ct_enriched", ":"), ("motif_shared_core", "--")):
        b = arm(res, MR)
        if b:
            ax.axvline(b["ood"], color=GREY, lw=1, ls=ls)
            refs.append((b["ood"], res))
    y = 0
    ticks, labels = [], []
    for fam, (base, variants, colour) in FAMILIES.items():
        vals = [(v, arm(v)) for v in variants]
        vals = [(v, s["ood"]) for v, s in vals if s]
        if not vals:
            continue
        xs = [x for _, x in vals]
        ax.scatter(xs, [y] * len(xs), s=26, c=colour, alpha=0.85, zorder=3)
        ax.plot([min(xs), max(xs)], [y, y], color=colour, lw=1.2, alpha=0.45, zorder=2)
        if base:
            b = arm(base, MR)
            if b:
                ax.scatter(
                    [b["ood"]], [y], s=90, facecolors="none", edgecolors="black", lw=1.4, zorder=4
                )
        best = max(vals, key=lambda t: t[1])
        ax.annotate(
            best[0],
            (best[1], y),
            textcoords="offset points",
            xytext=(7, 0),
            fontsize=7,
            va="center",
            color=colour,
        )
        ticks.append(y)
        labels.append(f"{fam}  (n={len(vals)})")
        y -= 1
    ax.set_yticks(ticks, labels)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.13)  # room for the best-member labels
    ax.set_ylim(min(ticks) - 0.9, max(ticks) + 0.6)
    for x, res in refs:  # labelled inside the axes, not off the top of the figure
        ax.text(x, min(ticks) - 0.85, f" {res}", rotation=90, fontsize=6.5, color=GREY, va="bottom")
    ax.set_xlabel("designed-set r  (stacked, 3 seeds, 30k)")
    ax.set_title(
        "Within-family spread is often larger than between-family\n"
        "open circle = that family's default; label = best member; grey lines = the two "
        "motif reservoirs",
        loc="left",
    )
    fig.savefig(out / "fig2_within_family_spread.png")
    plt.close(fig)
    print("fig2 written")


# --------------------------------------------------------------------------- fig 3

# (arm, sites-per-sequence midpoint). The base plants 3-7.
DENSITY = {
    "motif_ct_enriched": [
        ("mot_ct_n1_2", 1.5),
        (None, 5.0),
        ("mot_ct_n8_15", 11.5),
        ("mot_ct_n15_25", 20.0),
        ("mot_ct_n25_40", 32.5),
        ("mot_ct_n40_60", 50.0),
    ],
    "motif_shared_core": [
        (None, 5.0),
        ("mot_sc_n8_15", 11.5),
        ("mot_sc_n15_25", 20.0),
    ],
}


def fig3(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, key in zip(axes, ("ood", "genomic"), strict=False):
        for (base, rungs), colour in zip(DENSITY.items(), (BLUE, GREEN), strict=False):
            xs, ys = [], []
            for name, x in rungs:
                s = arm(base, MR) if name is None else arm(name)
                if s:
                    xs.append(x)
                    ys.append(s[key])
            if xs:
                ax.plot(xs, ys, "o-", color=colour, lw=1.6, ms=5, label=base)
        ax.axvspan(25, 60, color=GREY, alpha=0.08)
        ax.set_xlabel("planted sites per 200 bp (midpoint of range)")
        ax.set_title(f"{PRETTY[key]} r")
    axes[0].set_ylabel("stacked r")
    axes[0].legend(fontsize=8)
    axes[0].text(
        42, axes[0].get_ylim()[0] + 0.01, "overlap\nforced", fontsize=7, color=GREY, ha="center"
    )
    fig.suptitle(
        "Planting density is the one reservoir parameter that moves the designed endpoint",
        y=1.03,
        fontsize=10.5,
    )
    fig.savefig(out / "fig3_density_ladder.png")
    plt.close(fig)
    print("fig3 written")


# --------------------------------------------------------------------------- fig 4


def inactive_fraction(weights: str) -> float:
    """Share of planted sites drawn from classes not expected to be active in K562."""
    tot, bad = 0.0, 0.0
    for part in weights.split(","):
        k, _, v = part.partition("=")
        w = float(v)
        tot += w
        if k.strip() in ("hepg2_only", "decoy", "other"):
            bad += w
    return bad / tot if tot else 0.0


def fig4(out: Path) -> None:
    try:
        from variants2 import VARIANTS as V2
    except Exception as e:  # pragma: no cover - figure is optional
        print(f"fig4 skipped: {e}")
        return
    pts = []
    for name, _strategy, params, _why in V2:
        w = params.get("tier_weights")
        s = arm(name)
        if w and s:
            pts.append(
                (inactive_fraction(str(w)), s["ood"], name, params.get("tier_scope", "site"))
            )
    if len(pts) < 4:
        print("fig4: too few tier arms")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for scope, colour, marker in (("site", BLUE, "o"), ("sequence", VERM, "s")):
        g = [p for p in pts if p[3] == scope]
        if g:
            ax.scatter(
                [p[0] for p in g],
                [p[1] for p in g],
                s=42,
                c=colour,
                marker=marker,
                label=f"per-{scope} mixing",
                zorder=3,
            )
    b = arm("motif_ct_enriched", MR)
    if b:
        ax.axhline(b["ood"], color="black", lw=1.2, ls="--")
        ax.text(0.01, b["ood"] + 0.004, "motif_ct_enriched: the untiered reservoir", fontsize=7.5)
    # The control carries the argument: tier_active_only plants NOTHING the annotation calls
    # inactive and still loses most of the gap, so the cost is not the inactive content.
    ctrl = next((p for p in pts if p[2] == "tier_active_only"), None)
    if ctrl and b:
        ax.annotate(
            f"tier_active_only\n0% inactive, still {ctrl[1] - b['ood']:+.3f}",
            (ctrl[0], ctrl[1]),
            textcoords="offset points",
            xytext=(14, -6),
            fontsize=7.5,
            color=BLUE,
            arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.9},
        )
    x = np.array([p[0] for p in pts])
    yv = np.array([p[1] for p in pts])
    if len(x) > 2 and x.std() > 0:
        k = np.polyfit(x, yv, 1)
        xs = np.linspace(0, x.max(), 20)
        ax.plot(xs, np.polyval(k, xs), color=GREY, lw=1.2)
        ax.text(
            0.98,
            0.06,
            f"no dose-response: slope {k[0]:+.2f} r per unit inactive fraction,\n"
            f"Pearson {np.corrcoef(x, yv)[0, 1]:+.2f} over {len(x)} arms",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=GREY,
        )
    ax.set_xlabel("fraction of planted sites annotated inactive in K562 (incl. decoys)")
    ax.set_ylabel("designed-set r")
    ax.set_title(
        "Every tiered arm loses, but HOW MUCH inactive content does not predict how much.\n"
        "The cost is partitioning the vocabulary, not the inactive sites themselves.",
        loc="left",
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(out / "fig4_tier_cost.png")
    plt.close(fig)
    print("fig4 written")


# --------------------------------------------------------------------------- fig 5

A43 = REPO / "outputs/acq43"
A44 = REPO / "outputs/acq44"


def _one(root: Path, name: str) -> dict[str, float] | None:
    d = root / name
    if not d.is_dir():
        return None
    kids = [k for k in sorted(d.iterdir()) if k.is_dir()]
    return stacked(kids[0] if len(kids) == 1 else d)


def fig5(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # (a) candidate-pool size, rule held at epistemic, with random as the required null.
    ax = axes[0]
    caps = [(50_000, "cap50k_epi"), (500_000, "cap500k_epi"), (970_000, "cap970k_epi")]
    xs, ys = [], []
    for c, n in caps:
        s = _one(A43, n)
        if s:
            xs.append(c)
            ys.append(s["ood"])
    if xs:
        ax.plot(xs, ys, "o-", color=BLUE, lw=1.8, ms=6, label="epistemic")
    for n, lab, colour in (
        ("cap970k_rand", "random @970k", VERM),
        ("union_epi", "epistemic @1.97M union", SKY),
        ("union_rand", "random @1.97M union", ORANGE),
    ):
        s = _one(A43, n)
        if s:
            ax.scatter(
                [1_970_000 if "union" in n else 970_000],
                [s["ood"]],
                s=55,
                c=colour,
                label=lab,
                zorder=4,
            )
    ax.set_xscale("log")
    ax.set_xlabel("candidate sequences scored (--cap)")
    ax.set_ylabel("designed-set r")
    ax.set_title("(a) more candidates, to a point")
    ax.legend(fontsize=7.5, loc="lower right")

    # (b) the rules, all at the same reservoir, recipe, seeds and cap.
    ax = axes[1]
    # Colour encodes the RESERVOIR and position encodes the rule: colouring per rule as
    # well leaves the legend describing one thing and the bars another.
    rules = [
        ("random", "random"),
        ("epistemic", "epistemic"),
        ("activity_normalized", "activity-normalised"),
        ("expected_activity", "expected activity"),
        ("diversity_kmeans", "diversity (k-means)"),
    ]
    width, got, missing = 0.38, False, []
    for k, (res, off, colour) in enumerate(
        (
            ("motif_shared_core", -width / 2, BLUE),
            ("motif_ct_enriched", width / 2, GREEN),
        )
    ):
        xs, ys = [], []
        for i, (m, _lab) in enumerate(rules):
            s = _one(A44, f"{res}_{m}")
            if s:
                xs.append(i + off)
                ys.append(s["ood"])
                got = True
            else:
                missing.append(i + off)
        if xs:
            ax.bar(
                xs,
                ys,
                width=width,
                color=colour,
                edgecolor="black",
                linewidth=0.4,
                label=res.replace("motif_", ""),
            )
    if got:
        ax.set_xticks(range(len(rules)), [lab for _, lab in rules], rotation=20, ha="right")
        ax.set_ylabel("designed-set r")
        ax.set_ylim(0.55, None)
        lo = ax.get_ylim()[0]
        for x in missing:
            ax.text(x, lo + 0.004, "running", rotation=90, fontsize=6.5, ha="center", color=GREY)
        ax.legend(fontsize=7.5)
    else:
        note(ax, "acquisition rules still training")
    ax.set_title("(b) which rule, at matched everything")

    fig.suptitle(
        "Acquisition: the pool it selects from matters as much as the rule", y=1.04, fontsize=10.5
    )
    fig.savefig(out / "fig5_acquisition.png")
    plt.close(fig)
    print("fig5 written")


# --------------------------------------------------------------------------- fig 6 / 7

CT = REPO / "outputs/celltype"
MODES6 = ["agnostic", "onehot", "embed", "text", "text_shuffled"]


def _cd(prefix: str, mode: str, seeds=(0, 1, 2), hold: str | None = None) -> list[dict]:
    out = []
    for s in seeds:
        tail = f"_hold{hold}" if hold else ""
        f = CT / f"{prefix}_{mode}{tail}_s{s}" / "metrics.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def _mean_sec(runs: list[dict], sec: str) -> tuple[float, float]:
    per = []
    for r in runs:
        v = [
            x["r"]
            for x in r.get(sec, {}).values()
            if isinstance(x.get("r"), (int, float)) and math.isfinite(x["r"])
        ]
        if v:
            per.append(sum(v) / len(v))
    if not per:
        return float("nan"), 0.0
    return float(np.mean(per)), float(np.std(per))


def fig6(out: Path) -> None:
    secs = [
        ("genomic_test", "genomic\nactivity"),
        ("designed_test" if (CT / "x").exists() else "designed", "designed\nactivity"),
        ("genomic_test_differences", "genomic\nCELL DIFFERENCE"),
        ("designed_differences", "designed\nCELL DIFFERENCE"),
    ]
    data = {m: _cd("cd46", m) for m in MODES6}
    if not any(data.values()):
        print("fig6: no cd46 runs")
        return
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    colours = [GREY, BLUE, GREEN, ORANGE, VERM]
    w = 0.16
    for i, (m, colour) in enumerate(zip(MODES6, colours, strict=False)):
        mu = [_mean_sec(data[m], s)[0] for s, _ in secs]
        sd = [_mean_sec(data[m], s)[1] for s, _ in secs]
        xs = np.arange(len(secs)) + (i - 2) * w
        mu = [0 if not math.isfinite(v) else v for v in mu]
        ax.bar(
            xs,
            mu,
            width=w,
            color=colour,
            label=m,
            yerr=sd,
            capsize=1.5,
            error_kw={"lw": 0.7},
            edgecolor="black",
            linewidth=0.3,
        )
    for j, (sec, _) in enumerate(secs):
        if not math.isfinite(_mean_sec(data["agnostic"], sec)[0]):
            ax.text(j - 2 * w, 0.02, "n/a", rotation=90, fontsize=6.5, ha="center", color=GREY)
    ax.set_xticks(np.arange(len(secs)), [lab for _, lab in secs], fontsize=8)
    ax.set_ylabel("r (mean over cells and 3 seeds)")
    ax.set_title(
        "The cell channel is worth a great deal; what you put IN it is worth nothing.\n"
        "A one-hot ID matches a real description, and matches a deliberately WRONG one.",
        loc="left",
    )
    ax.legend(fontsize=7.5, ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(out / "fig6_conditioning_gate.png")
    plt.close(fig)
    print("fig6 written")


def fig7(out: Path) -> None:
    """Held-out cell: ct45's full LOCO grid, plus cd47's clean SK-N-SH rerun."""
    modes = ["agnostic", "text", "text_shuffled"]
    cells = ["K562", "HepG2", "SK-N-SH", "A549", "GM12878", "HCT116", "Jurkat"]
    got: dict[tuple[str, str], float] = {}
    for c in cells:
        for m in modes:
            runs = [r for r in _cd("ct45", m, (0, 1), hold=c) if r.get("diverged_epoch") is None]
            v = [
                r["genomic_test"][c]["r"]
                for r in runs
                if c in r.get("genomic_test", {})
                and r["genomic_test"][c].get("held_out")
                and math.isfinite(r["genomic_test"][c].get("r", float("nan")))
            ]
            if v:
                got[(c, m)] = float(np.mean(v))
    cd47 = {
        m: [
            r["genomic_test"]["SK-N-SH"]["r"]
            for r in _cd("cd47", m, (0, 1, 2), hold="SK-N-SH")
            if r.get("diverged_epoch") is None and "SK-N-SH" in r.get("genomic_test", {})
        ]
        for m in modes
    }
    if not got and not any(cd47.values()):
        print("fig7: no held-out runs")
        return
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    colours = {"agnostic": GREY, "text": BLUE, "text_shuffled": VERM}
    w = 0.26
    present = [c for c in cells if any((c, m) in got for m in modes)]
    # Positions tracked explicitly: the clean rerun sits in its own slot after a gap, so
    # the tick positions are not a plain arange over the labels.
    centres = list(range(len(present)))
    labels = list(present)
    for i, m in enumerate(modes):
        xs = np.array(centres, dtype=float) + (i - 1) * w
        ys = [got.get((c, m), 0.0) for c in present]
        ax.bar(xs, ys, width=w, color=colours[m], label=m, edgecolor="black", linewidth=0.3)
        # A missing bar here is a run that DIVERGED at lr 0.005 and was dropped, not a
        # measured zero. Say so on the axis rather than leaving a gap to be misread.
        for x, c in zip(xs, present, strict=False):
            if (c, m) not in got:
                ax.text(x, 0.012, "diverged", rotation=90, fontsize=6, ha="center", color=GREY)
    if any(cd47.values()):
        x0 = len(present) + 0.5
        for i, m in enumerate(modes):
            if cd47[m]:
                ax.bar(
                    x0 + (i - 1) * w,
                    float(np.mean(cd47[m])),
                    width=w,
                    color=colours[m],
                    hatch="//",
                    edgecolor="black",
                    linewidth=0.3,
                )
        centres.append(x0)
        labels.append("SK-N-SH\nclean rerun")
    ax.set_xticks(centres, labels, fontsize=7.5)
    ax.set_ylabel("r on the HELD-OUT cell")
    ax.set_title(
        "A cell description does not reach a cell type we have no measurements for.\n"
        "No conditioning at all is as good -- so every cell we intend to predict needs its "
        "own oligos.",
        loc="left",
    )
    ax.legend(fontsize=8)
    fig.savefig(out / "fig7_celltype_transfer.png")
    plt.close(fig)
    print("fig7 written")


def fig8(out: Path) -> None:
    """dp48: how much measured data a data-poor cell actually needs, and bought how."""
    files = sorted(CT.glob("dp48_*/panel.json")) + sorted(CT.glob("dp48_*/metrics.json"))
    if not files:
        print("fig8: dp48 has not written results yet")
        return
    cells = sorted({f.parent.name.replace("dp48_", "") for f in files})
    # Two rows, because the two endpoints move in OPPOSITE directions and a figure showing
    # only r_all would say "buy data, it helps" and hide that the same data costs
    # specificity. The 0-shot value is drawn as a horizontal line rather than a point at
    # x=0, which a log axis cannot show.
    fig, axes = plt.subplots(
        2, len(cells), figsize=(4.1 * len(cells), 6.2), squeeze=False, sharex=True
    )
    colours = [GREY, BLUE, GREEN, PURPLE, VERM, ORANGE]
    for col, cell in enumerate(cells):
        f = next((p for p in files if p.parent.name == f"dp48_{cell}"), None)
        if f is None:
            continue
        rec = json.loads(f.read_text())
        rows = rec.get("results", rec if isinstance(rec, list) else [])
        for row, key, ylab in (
            (0, "r_all", "r on the target cell"),
            (1, "r_specific", "r on the target-minus-reference DIFFERENCE"),
        ):
            ax = axes[row][col]
            by: dict[str, list[tuple[int, float]]] = {}
            zero = None
            for r in rows:
                v = r.get(key)
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    continue
                if r.get("strategy") == "none" or r.get("shots") in (0, None):
                    zero = v
                    continue
                if "strategy" in r and "shots" in r:
                    by.setdefault(r["strategy"], []).append((r["shots"], v))
            for (s, pts), colour in zip(sorted(by.items()), colours, strict=False):
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colour, ms=4, label=s)
            if zero is not None:
                ax.axhline(zero, color="black", lw=1.2, ls="--")
                ax.text(
                    1100, zero, " no target data at all", fontsize=7, va="bottom", color="black"
                )
            ax.set_xscale("log")
            if row == 1:
                ax.set_xlabel("sequences measured in this cell")
            if col == 0:
                ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(f"{cell}  ({'activity' if row == 0 else 'specificity'})", fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(
        "A third cell line needs ~1k sequences for most of its activity gain -- and the "
        "same data makes SPECIFICITY worse",
        y=1.0,
        fontsize=10.5,
    )
    fig.savefig(out / "fig8_datapoor_allocation.png")
    plt.close(fig)
    print("fig8 written")


# --------------------------------------------------------------------------- fig 9


def fig9(out: Path) -> None:
    log = sorted((REPO / "outputs/design47/logs").glob("dm47_*.log"))
    if not log:
        print("fig9: no dm47 log")
        return
    pat = re.compile(
        r"^\s*(\w+)\s+n_(?:steps|rounds)=(\d+) guide\s+([\d.]+) \| oracle\s+([\d.]+) \| "
        r"gap\s+([+\-\d.]+) \| p90\s+([\d.]+)"
    )
    rows: dict[str, list[tuple[int, float, float, float]]] = {}
    for line in log[-1].read_text().splitlines():
        m = pat.match(line)
        if m:
            rows.setdefault(m.group(1), []).append(
                (int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(6)))
            )
    if not rows:
        print("fig9: nothing parsed")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    colours = {"fastseqprop": VERM, "sim_anneal": BLUE, "adalead": GREEN}
    for name, pts in rows.items():
        pts.sort()
        c = colours.get(name, GREY)
        axes[0].plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "o--",
            color=c,
            ms=5,
            label=f"{name} (guide)",
            alpha=0.55,
        )
        axes[0].plot(
            [p[0] for p in pts], [p[2] for p in pts], "o-", color=c, ms=5, label=f"{name} (oracle)"
        )
        axes[1].plot([p[1] for p in pts], [p[2] for p in pts], "o-", color=c, ms=5, label=name)
        for p in pts:
            axes[1].annotate(
                str(p[0]),
                (p[1], p[2]),
                textcoords="offset points",
                xytext=(4, -8),
                fontsize=6,
                color=c,
            )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("optimisation budget (steps or rounds)")
    axes[0].set_ylabel("mean predicted activity")
    axes[0].set_title("(a) the guide keeps rising; the oracle does not")
    axes[0].legend(fontsize=6.5, ncols=2)
    lim = axes[1].get_xlim()
    axes[1].plot(lim, lim, color=GREY, lw=1, ls=":")
    axes[1].set_xlim(lim)
    axes[1].set_xlabel("guide-predicted activity")
    axes[1].set_ylabel("oracle-measured activity")
    axes[1].set_title("(b) dotted line = perfect calibration")
    axes[1].legend(fontsize=7.5)
    fig.suptitle(
        "Design methods at matched compute: the overshoot is not a fixable offset",
        y=1.04,
        fontsize=10.5,
    )
    fig.savefig(out / "fig9_design_budget.png")
    plt.close(fig)
    print("fig9 written")


# --------------------------------------------------------------------------- fig 11

A50 = REPO / "outputs/acq50"
RULES11 = [
    ("random", "random", GREY),
    ("epistemic", "epistemic", BLUE),
    ("activity_normalized", "activity-norm.", GREEN),
    ("expected_activity", "expected act.", VERM),
    ("diversity_kmeans", "diversity", ORANGE),
]
# Ordered by Arm A's designed-endpoint rank, so the x axis IS reservoir quality and a rule
# whose behaviour depends on quality shows as a sloped line rather than a scramble.
RES11 = [
    ("dinuc_shuffle", "dinuc"),
    ("random", "random"),
    ("gc_matched", "gc_match"),
    ("genomic", "genomic"),
    ("evoaug_paper2023", "evoaug"),
    ("mutagenesis", "mutagen"),
    ("motif_ct_enriched", "motif_ct"),
    ("motif_shared_core", "motif_sc"),
]


def fig11(out: Path) -> None:
    """Is a rule's behaviour a property of the rule, or of the rule x reservoir pair?"""
    data: dict[str, dict] = {}
    for res, _lab in RES11:
        f = A50 / f"{res}.json"
        if f.exists():
            data[res] = json.loads(f.read_text())
    if len(data) < 3:
        print(f"fig11: only {len(data)} reservoirs profiled yet")
        return

    def pick_bank(rec: dict, prefer_own: str) -> dict | None:
        own = f"bench38_ln_e500_wd2_30k_c20_{prefer_own}"
        if own in rec["banks"]:
            return rec["banks"][own]
        return rec["banks"].get("bench38_ln_e500_wd2_30k_c20")

    present = [(r, lab) for r, lab in RES11 if r in data]
    xs = np.arange(len(present))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), gridspec_kw={"wspace": 0.3})

    # (a) activity shift, in pool-sd units so pools with different label scales compare.
    # (b) k-mer spread of the selection over the pool's: does the rule narrow the corpus.
    for ax, key, ylab, title in (
        (
            axes[0],
            "label_z_vs_pool",
            "selected activity  (pool sd)",
            "(a) what each rule buys, by reservoir quality",
        ),
        (
            axes[1],
            "kmer_spread_ratio",
            "4-mer spread / pool",
            "(b) does the rule narrow sequence space?",
        ),
    ):
        for m, lab, colour in RULES11:
            ys = []
            for r, _ in present:
                b = pick_bank(data[r], r)
                v = (b or {}).get("rules", {}).get(m, {}).get(key)
                ys.append(v if isinstance(v, (int, float)) else np.nan)
            ax.plot(xs, ys, "o-", color=colour, ms=5, lw=1.6, label=lab)
        ax.axhline(0.0 if key == "label_z_vs_pool" else 1.0, color=GREY, lw=1, ls="--")
        ax.set_xticks(xs, [lab for _, lab in present], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
    axes[0].legend(fontsize=7, ncols=2)

    # (c) the decomposition: matched bank minus common bank, on the activity shift. A
    # non-zero value means the rule behaves differently because the MODEL changed, not
    # because the candidates did.
    ax = axes[2]
    any_pair = False
    for m, lab, colour in RULES11:
        ys = []
        for r, _ in present:
            rec = data[r]
            own = rec["banks"].get(f"bench38_ln_e500_wd2_30k_c20_{r}")
            com = rec["banks"].get("bench38_ln_e500_wd2_30k_c20")
            if own and com:
                a_ = own.get("rules", {}).get(m, {}).get("label_z_vs_pool")
                b_ = com.get("rules", {}).get(m, {}).get("label_z_vs_pool")
                ys.append(
                    a_ - b_
                    if isinstance(a_, (int, float)) and isinstance(b_, (int, float))
                    else np.nan
                )
                any_pair = True
            else:
                ys.append(np.nan)
        ax.plot(xs, ys, "o-", color=colour, ms=5, lw=1.6, label=lab)
    ax.axhline(0.0, color=GREY, lw=1, ls="--")
    ax.set_xticks(xs, [lab for _, lab in present], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("matched bank - common bank")
    ax.set_title("(c) is it the candidates or the ranker?", fontsize=9.5)
    if not any_pair:
        note(ax, "no reservoir has both banks yet")

    fig.suptitle(
        "Acquisition x reservoir, profiled without training: parallel lines mean the rule "
        "axis can be explored on one reservoir",
        y=1.05,
        fontsize=10.5,
    )
    fig.savefig(out / "fig11_acq_pool_interaction.png")
    plt.close(fig)
    print("fig11 written")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--only", default=None, help="comma-separated figure numbers")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    todo = {
        "1": fig1,
        "2": fig2,
        "3": fig3,
        "4": fig4,
        "5": fig5,
        "6": fig6,
        "7": fig7,
        "8": fig8,
        "9": fig9,
        "11": fig11,
    }
    want = a.only.split(",") if a.only else list(todo)
    for k in want:
        fn = todo.get(k.strip())
        if fn is None:
            continue
        try:
            fn(out)
        except Exception as e:
            print(f"fig{k} FAILED: {type(e).__name__}: {e}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
