"""Which of the measured eval sets are worth putting on a slide?

The last meeting asked for exactly this: measure an exhaustive panel, report a few
highlights that cover the important use cases, are not easy enough to saturate, and are
reasonably independent of each other. Every arm already stores predictions for 18 eval
sets, so the question is answerable from data we have rather than by argument.

Three quantities per endpoint, all computed across the candidate corpora only (a corpus
that teaches the student nothing drags every endpoint down together, and including those
makes everything look both hard and correlated):

  DISCRIMINATION  between-corpus sd / within-corpus seed sd. This is the one that matters
                  and the one usually left implicit. An endpoint whose spread across
                  corpora is no larger than the spread across re-runs of the SAME corpus
                  cannot rank anything, however interesting it sounds.
  SATURATION      mean r. An endpoint everything scores 0.95 on has no headroom left.
  REDUNDANCY      correlation with the other endpoints, across corpora. Two endpoints that
                  rank the corpora identically are one endpoint and one slide.

Besides the 18 stored LEVEL endpoints this also builds the DERIVED EFFECT endpoints, which
the Nov 19 analysis found to be the most informative new axis: every perturbation set
(substitution, insertion, deletion, translocation, inversion, dinucleotide shuffle) is
row-aligned with the genomic set, and snv_ref/snv_alt are a matched pair, so

    effect(X) = r( pred_X - pred_genomic , oracle_X - oracle_genomic )

asks whether the student predicts what the perturbation DID, rather than whether it can
still predict a perturbed sequence's level -- a much harder and much less redundant thing.

Finally it runs a greedy decorrelated selection: take the most discriminating endpoint,
then repeatedly add whichever endpoint has the lowest maximum correlation with those
already chosen. That is the shortlist to report.

    python scratchpad/nov25/eval_panel_selection.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "orchestration"))

VAR = REPO / "outputs/var41"
MR = REPO / "outputs/menu_retrain_v0"
PREFIX = "bench38_ln_e500_wd2_30k_c20"

# Perturbation sets that are row-aligned with `genomic`, so an effect endpoint exists.
PERTURB = (
    "dinuc_shuffle",
    "sub_low",
    "sub_med",
    "sub_high",
    "ins_low",
    "ins_med",
    "ins_high",
    "del_low",
    "del_med",
    "del_high",
    "translocation",
    "inversion",
)
BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6b6b6b"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def arm_members(d: Path) -> list[dict[str, float]] | None:
    """Per-member r on every level and effect endpoint, for one arm directory."""
    lab_p = d / "labels.npz"
    metas = sorted(glob.glob(str(d / "*_meta.json")))
    if not metas or not lab_p.exists():
        return None
    lab = np.load(lab_p, allow_pickle=True)
    sets = [k[len("oracle_") :] for k in lab.files if k.startswith("oracle_")]
    Y = {s: np.asarray(lab[f"oracle_{s}"], float).ravel() for s in sets}

    out = []
    for m in metas:
        z = np.load(m.replace("_meta.json", ".npz"), allow_pickle=True)
        if "test_pred_genomic" not in z.files:
            continue
        P = {
            s: np.asarray(z[f"test_pred_{s}"], float).ravel()
            for s in sets
            if f"test_pred_{s}" in z.files
        }
        row: dict[str, float] = {s: pearson(P[s], Y[s]) for s in P}
        # Derived effect endpoints. Only where the lengths actually match -- an eval set
        # that is not a perturbation OF the genomic rows would otherwise be differenced
        # against unrelated sequences and produce a confident, meaningless number.
        if "genomic" in P:
            for s in PERTURB:
                if s in P and len(P[s]) == len(P["genomic"]) and len(Y[s]) == len(Y["genomic"]):
                    row[f"D:{s}"] = pearson(P[s] - P["genomic"], Y[s] - Y["genomic"])
        if "snv_ref" in P and "snv_alt" in P and len(P["snv_ref"]) == len(P["snv_alt"]):
            row["D:snv"] = pearson(P["snv_alt"] - P["snv_ref"], Y["snv_alt"] - Y["snv_ref"])
        out.append(row)
    return out or None


def collect(min_genomic: float) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Arm means and within-arm seed sds, restricted to candidate corpora."""
    dirs = sorted(VAR.glob(f"{PREFIX}_*")) + [
        MR / f"{PREFIX}_{n}"
        for n in (
            "genomic",
            "random",
            "random_gc_sweep",
            "gc_matched",
            "dinuc_shuffle",
            "mutagenesis",
            "evoaug",
            "motif_shared_core",
            "motif_ct_enriched",
            "motif_syntax_core",
            "accessible_tiles_k562",
        )
    ]
    names, means, sds = [], [], []
    keys: list[str] | None = None
    for d in dirs:
        got = arm_members(d)
        if not got or len(got) < 2:
            continue
        if keys is None:
            keys = sorted(got[0])
        M = np.array([[g.get(k, np.nan) for k in keys] for g in got], float)
        gi = keys.index("genomic")
        if not np.isfinite(np.nanmean(M[:, gi])) or np.nanmean(M[:, gi]) < min_genomic:
            continue
        names.append(d.name[len(PREFIX) + 1 :] or d.name)
        means.append(np.nanmean(M, axis=0))
        sds.append(np.nanstd(M, axis=0, ddof=1))
    return names, (keys or []), np.array(means), np.array(sds)


def greedy(
    keys: list[str],
    disc: np.ndarray,
    mean_r: np.ndarray,
    C: np.ndarray,
    k: int = 5,
    min_disc: float = 2.0,
    max_r: float = 0.90,
) -> list[int]:
    """Most discriminating first, then lowest maximum correlation with what is chosen.

    Both filters come straight from the brief: an endpoint must be able to out-rank a
    re-run (`min_disc`) and must not already be saturated (`max_r`). Without the second one
    the selection reaches for gc_sweep, which is decorrelated from everything precisely
    because every corpus scores 0.92 on it and the ordering is close to noise.
    """
    ok = [
        j
        for j in range(len(keys))
        if np.isfinite(disc[j]) and disc[j] >= min_disc and mean_r[j] <= max_r
    ]
    if not ok:
        return []
    order = [max(ok, key=lambda j: disc[j])]
    while len(order) < k:
        rest = [j for j in ok if j not in order]
        if not rest:
            break
        # Lowest worst-case correlation with the current set; ties go to the more
        # discriminating endpoint.
        best = min(
            rest,
            key=lambda j: (
                np.nanmax(np.abs([C[j, i] for i in order])) - 0.02 * disc[j] / float(max(disc))
            ),
        )
        order.append(best)
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--min-genomic", type=float, default=0.85)
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    names, keys, M, S = collect(a.min_genomic)
    if len(M) < 8:
        print(f"only {len(M)} candidate corpora; not enough to measure discrimination")
        return 1
    between = np.nanstd(M, axis=0, ddof=1)
    within = np.nanmean(S, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = between / within
    mean_r = np.nanmean(M, axis=0)
    C = np.corrcoef(np.ma.masked_invalid(M).filled(np.nanmean(M, axis=0)).T)

    print(
        f"{len(names)} candidate corpora x {len(keys)} endpoints (genomic r >= {a.min_genomic})\n"
    )
    hdr = f"{'endpoint':<20}{'mean r':>9}{'between':>10}{'within':>9}{'discrim':>9}"
    print(hdr)
    print("-" * len(hdr))
    for i in np.argsort(-np.nan_to_num(disc)):
        print(f"{keys[i]:<20}{mean_r[i]:>9.4f}{between[i]:>10.4f}{within[i]:>9.4f}{disc[i]:>9.2f}")

    pick = greedy(keys, disc, mean_r, C, k=5)
    print("\nShortlist: discrimination >= 2.0, mean r <= 0.90, then greedily decorrelated")
    for n, i in enumerate(pick, 1):
        worst = max((abs(C[i, j]) for j in pick if j != i), default=0.0)
        print(
            f"  {n}. {keys[i]:<18} discrim {disc[i]:5.2f}  mean r {mean_r[i]:.3f}  "
            f"max |corr| with the rest {worst:.2f}"
        )

    # --------------------------------------------------------------- figure
    fig = plt.figure(figsize=(12.6, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.85], wspace=0.46)

    ax = fig.add_subplot(gs[0, 0])
    is_eff = np.array([k.startswith("D:") for k in keys])
    for m, colour, lab in (
        (~is_eff, BLUE, "level (predict activity)"),
        (is_eff, VERM, "effect (predict the change)"),
    ):
        ax.scatter(mean_r[m], disc[m], s=26, c=colour, label=lab, zorder=3)
    for i in pick:
        ax.annotate(
            keys[i].replace("D:", "Δ"),
            (mean_r[i], disc[i]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color=GREEN,
            fontweight="bold",
        )
    ax.axhline(1.0, color=GREY, lw=1, ls="--")
    ax.text(ax.get_xlim()[0], 1.05, " cannot out-rank a re-run", fontsize=7, color=GREY)
    ax.set_xlabel("mean r across candidate corpora  (saturation ->)")
    ax.set_ylabel("discrimination  (between-corpus sd / seed sd)")
    ax.set_title("(a) only a few endpoints can rank anything", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    o = np.argsort(-np.nan_to_num(disc))
    im = ax.imshow(C[np.ix_(o, o)], cmap="RdBu_r", vmin=-1, vmax=1)
    short = [keys[i].replace("D:", "Δ") for i in o]
    ax.set_xticks(range(len(short)), short, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(short)), short, fontsize=5.5)
    ax.set_title("(b) how the endpoints rank corpora\n(ordered by discrimination)", fontsize=9.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8, pad=0.03)

    ax = fig.add_subplot(gs[0, 2])
    X = np.ma.masked_invalid(M).filled(np.nanmean(M, axis=0))
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ev = np.linalg.svd(X, compute_uv=False) ** 2
    cum = np.cumsum(ev) / ev.sum()
    ax.plot(range(1, len(cum) + 1), cum, "o-", color=BLUE, ms=4)
    ax.axhline(0.9, color=GREY, lw=1, ls="--")
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    ax.axvline(n90, color=GREY, lw=1, ls=":")
    ax.annotate(
        f"{n90} components carry 90%\nof the variance across\nall {len(keys)} endpoints",
        xy=(0.95, 0.12),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color=GREY,
    )
    ax.set_xlim(0.5, min(12, len(cum)) + 0.5)
    ax.set_xlabel("principal component")
    ax.set_ylabel("cumulative variance")
    ax.set_title("(c) how many real dimensions", fontsize=9.5)

    fig.suptitle(
        f"Eval panel: {len(keys)} endpoints measured on {len(names)} candidate corpora, "
        "and the few worth reporting",
        y=1.04,
        fontsize=10.5,
    )
    fig.savefig(out / "fig10_eval_panel_selection.png")
    plt.close(fig)
    print(f"\n-> {out / 'fig10_eval_panel_selection.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
