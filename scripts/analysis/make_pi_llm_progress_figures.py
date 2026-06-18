"""PI-update figures: LLM-AutoResearch progress.

Data sources (all REAL, no synthetic numbers):
  - personas-vs-perf: best ensemble oracle-r by number of personas (Phase-0
    confirm, persona_combos "best subset by size", k562_genomic_d30000, 3 seeds).
  - round-progress / models-vs-perf: per-model JSON metrics from the LIVE Phase-1
    anchor pool (outputs/hp_step1_bakeoff_e100/k562_genomic_d30000) -> /tmp/pi_agg.json.
  - *_confirm variants: same two plots from the previous full-depth LLM runs
    (outputs/hp_llm_ablation_confirm_e100, 25 rounds, 75 models/persona/seed) ->
    /tmp/pi_agg_confirm.json. These are the byte-identical atoms now reused.
Metrics: val_pearson (chr-val) and per_set_metrics['genomic'] (held-out test).

Writes PNG+PDF to ~/Downloads/pi_update_figures/.
"""

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/Downloads/pi_update_figures")
AGG = "/tmp/pi_agg.json"

BLUE, RED, GREEN, ORANGE, PURPLE, GREY = (
    "#4C72B0",
    "#C44E52",
    "#55A868",
    "#DD8452",
    "#8172B3",
    "#7f7f7f",
)

plt.rcParams.update({"font.size": 11})


# --------------------------------------------------------------------------
# Plot 1 — ensemble performance vs number of personas
# --------------------------------------------------------------------------
def fig_personas_vs_perf():
    # "best subset by size" from Phase-0 confirm (persona_combos, 3 seeds).
    sizes = [1, 2, 3, 4, 5, 6]
    best_r = [0.7009, 0.7319, 0.7423, 0.7456, 0.7467, 0.7472]
    sd = [0.0130, 0.0135, 0.0140, 0.0146, 0.0138, 0.0132]
    labels = [
        "diverse",
        "+explore",
        "+exploit",
        "+default",
        "+critic",
        "+neutral",
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.errorbar(
        sizes,
        best_r,
        yerr=sd,
        marker="o",
        ms=9,
        lw=2.2,
        capsize=4,
        color=RED,
        mfc="white",
        mec=RED,
        mew=2,
        ecolor=GREY,
        zorder=3,
    )
    for x, y, lab in zip(sizes, best_r, labels):
        ax.annotate(
            lab, (x, y), textcoords="offset points", xytext=(8, -14), fontsize=9, color=GREY
        )

    # diminishing-returns shading: knee ~ size 3
    ax.axvspan(3, 6, color=GREEN, alpha=0.06)
    ax.annotate(
        "diminishing returns\n(knee ~3 personas)",
        (4.5, 0.7300),
        ha="center",
        fontsize=9.5,
        color=GREEN,
    )

    # delta annotations
    for i in range(1, len(sizes)):
        d = best_r[i] - best_r[i - 1]
        ax.annotate(
            f"+{d:.3f}",
            ((sizes[i] + sizes[i - 1]) / 2, (best_r[i] + best_r[i - 1]) / 2 + 0.0012),
            ha="center",
            fontsize=8.5,
            color=BLUE,
        )

    ax.set_xlabel("number of LLM personas in the ensemble")
    ax.set_ylabel("ensemble oracle correlation  (mean ± SD, 3 seeds)")
    ax.set_title(
        "More search personas help — but saturate quickly\n"
        "best persona subset at each size (Phase-0 confirm, D=30k)",
        fontsize=12,
    )
    ax.set_xticks(sizes)
    ax.set_ylim(0.695, 0.752)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/pi_llm_personas_vs_perf.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 2 — average model performance by optimization round
# --------------------------------------------------------------------------
PERSONAS = {
    "llm_exploit_nv1": ("exploit", RED),
    "llm_critic_nv0": ("critic", PURPLE),
    "llm_diverse_nv1": ("diverse", GREEN),
    "llm_explore_nv1": ("explore", ORANGE),
}


def _round_means(rows, key="genomic"):
    by_round = defaultdict(list)
    for r in rows:
        v = r["genomic"] if key == "genomic" else r["val"]
        if v is not None and np.isfinite(v):
            by_round[r["round"]].append(v)
    rounds = sorted(by_round)
    mean = np.array([np.mean(by_round[k]) for k in rounds])
    n = np.array([len(by_round[k]) for k in rounds])
    return np.array(rounds), mean, n


def fig_round_progress(
    agg, outname="pi_llm_round_progress", subtitle="live Phase-1, D=30k, 3 seeds"
):
    pm = agg["per_model"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for strat, (name, c) in PERSONAS.items():
        rows = pm.get(strat, [])
        if not rows:
            continue
        rounds, mean, n = _round_means(rows, key="genomic")
        if len(rounds) < 2:
            ax.scatter(rounds, mean, color=c, s=40, label=f"{name} (n={n.sum()})")
            continue
        ax.plot(
            rounds,
            mean,
            marker="o",
            ms=4,
            lw=1.4,
            color=c,
            alpha=0.85,
            label=f"{name} (n={n.sum()})",
        )
        # rolling-mean trend
        if len(rounds) >= 5:
            w = 5
            kern = np.ones(w) / w
            sm = np.convolve(mean, kern, mode="valid")
            ax.plot(rounds[w - 1 :], sm, lw=3, color=c, alpha=0.55)

    ax.set_xlabel("LLM-AutoResearch optimization round")
    ax.set_ylabel("mean model quality per round\n(held-out genomic test correlation)")
    ax.set_title(
        "Do later rounds propose better models?\n"
        f"per-round mean over proposed configs ({subtitle})",
        fontsize=12,
    )
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(title="persona", fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{outname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 3 — real performance vs number of models (live Phase 1)
# --------------------------------------------------------------------------
STRAT_STYLE = {
    "random": ("random", GREY, "-"),
    "optuna_tpe": ("optuna_tpe", BLUE, "-"),
    "optuna_cmaes": ("optuna_cmaes", "#6Fa8DC", "-"),
    "optuna_gp": ("optuna_gp", "#9FC5E8", "--"),
    "optuna_qmc": ("optuna_qmc", "#B6D7A8", "--"),
    "evo_explore": ("evo_explore", "#674EA7", "-"),
    "evo_batch": ("evo_batch", "#8E7CC3", "-"),
    "evo_single": ("evo_single", "#B4A7D6", "--"),
    "llm_exploit_nv1": ("LLM exploit", RED, "-"),
    "llm_diverse_nv1": ("LLM diverse", GREEN, "-"),
    "llm_critic_nv0": ("LLM critic", ORANGE, "-"),
    "llm_explore_nv1": ("LLM explore", "#E06666", "-"),
}


def _cumbest_one(rows, key):
    """Cumulative-best of `key` over a single search trajectory (mtime order)."""
    r = sorted(rows, key=lambda x: x["mtime"])
    vals = [x[key] for x in r if x[key] is not None and np.isfinite(x[key])]
    return np.maximum.accumulate(vals) if vals else np.array([])


def _cumbest_by_seed(rows, key):
    """Per-seed cumulative-best -> aligned mean and min/max across seeds vs model index."""
    bys = defaultdict(list)
    for x in rows:
        bys[x["seed"]].append(x)
    curves = [_cumbest_one(v, key) for v in bys.values()]
    curves = [c for c in curves if len(c)]
    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([])
    n = max(len(c) for c in curves)
    x = np.arange(1, n + 1)
    # at index i, average over seeds that have reached >= i+1 models
    mean = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for i in range(n):
        pts = [c[i] for c in curves if len(c) > i]
        mean[i] = np.mean(pts)
        lo[i] = np.min(pts)
        hi[i] = np.max(pts)
    return x, mean, lo, hi


def _cumbest_seed42(rows, key):
    r = [x for x in rows if x["seed"] == "seed42_0"]
    if not r:
        # fall back to whatever seed has the most rows
        bys = defaultdict(list)
        for x in rows:
            bys[x["seed"]].append(x)
        r = max(bys.values(), key=len)
    r = sorted(r, key=lambda x: x["mtime"])
    vals = [x[key] for x in r if x[key] is not None and np.isfinite(x[key])]
    cb = np.maximum.accumulate(vals)
    return np.arange(1, len(cb) + 1), cb


def fig_models_vs_perf(
    agg,
    outname="pi_llm_models_vs_perf",
    subtitle="Live Phase-1 search progress (D=30k, genomic, random acq.)",
    strat_style=STRAT_STYLE,
):
    pm = agg["per_model"]
    panels = [
        ("val", "best chr-val correlation so far\n(metric the search selects on)"),
        ("genomic", "best held-out genomic TEST correlation so far\n(unbiased generalization)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, (key, ylab) in zip(axes, panels):
        ymins, ymaxs = [], []
        for strat, (name, c, ls) in strat_style.items():
            rows = pm.get(strat, [])
            if not rows:
                continue
            x, cb = _cumbest_seed42(rows, key)
            if len(cb) == 0:
                continue
            ymins.append(cb.min())
            ymaxs.append(cb.max())
            is_llm = strat.startswith("llm_")
            ax.plot(
                x,
                cb,
                ls=ls,
                lw=2.8 if is_llm else 1.5,
                color=c,
                alpha=0.95 if is_llm else 0.6,
                marker="o" if (is_llm and len(x) <= 30) else None,
                ms=3,
                label=name,
                zorder=3 if is_llm else 2,
            )
        # autoscale to capture all data, with small padding
        lo, hi = min(ymins), max(ymaxs)
        pad = max(0.005, (hi - lo) * 0.06)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("number of models trained (search progress, seed 42)")
        ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=7.5, ncol=2, loc="lower right", framealpha=0.95)
    fig.suptitle(
        f"{subtitle}  — cumulative-best single model vs models evaluated",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{outname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


# LLM-only style map for the confirm-run trajectories (no algo/evo in that pool).
LLM_STYLE = {
    "llm_exploit_nv1": ("exploit", RED, "-"),
    "llm_critic_nv0": ("critic", PURPLE, "-"),
    "llm_diverse_nv1": ("diverse", GREEN, "-"),
    "llm_explore_nv1": ("explore", ORANGE, "-"),
}


def fig_models_vs_perf_multiseed(agg, outname, subtitle, strat_style=LLM_STYLE, n_seeds=3):
    """Cumulative-best vs models, mean across seeds + min-max band (per-seed runs)."""
    pm = agg["per_model"]
    panels = [
        ("val", "best chr-val correlation so far\n(metric the search selects on)"),
        ("genomic", "best held-out genomic TEST correlation so far\n(unbiased generalization)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, (key, ylab) in zip(axes, panels):
        ymins, ymaxs = [], []
        for strat, (name, c, ls) in strat_style.items():
            rows = pm.get(strat, [])
            if not rows:
                continue
            x, mean, lo, hi = _cumbest_by_seed(rows, key)
            if len(mean) == 0:
                continue
            ymins.append(np.nanmin(lo))
            ymaxs.append(np.nanmax(hi))
            ax.fill_between(x, lo, hi, color=c, alpha=0.13, zorder=2)
            ax.plot(x, mean, ls=ls, lw=2.6, color=c, label=name, zorder=3)
        lo_y, hi_y = min(ymins), max(ymaxs)
        pad = max(0.005, (hi_y - lo_y) * 0.06)
        ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_xlabel(f"number of models trained (search progress, mean of {n_seeds} seeds)")
        ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    fig.suptitle(
        f"{subtitle}  — cumulative-best vs models (mean of {n_seeds} seeds, band = seed min-max)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{outname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


AGG_CONFIRM = "/tmp/pi_agg_confirm.json"


def main():
    os.makedirs(OUT, exist_ok=True)
    agg = json.load(open(AGG))
    fig_personas_vs_perf()
    fig_round_progress(agg)
    fig_models_vs_perf(agg)
    written = ["pi_llm_personas_vs_perf", "pi_llm_round_progress", "pi_llm_models_vs_perf"]

    # Confirm-run (previous full-depth, 25-round) LLM trajectories — the data we are
    # now reusing as bake-off atoms. LLM-only pool; full depth where the live arms barely started.
    if os.path.exists(AGG_CONFIRM):
        aggc = json.load(open(AGG_CONFIRM))
        fig_round_progress(
            aggc,
            outname="pi_llm_round_progress_confirm",
            subtitle="previous full-depth runs, 25 rounds, D=30k, 3 seeds",
        )
        fig_models_vs_perf(
            aggc,
            outname="pi_llm_models_vs_perf_confirm",
            subtitle="Previous full-depth LLM runs (D=30k, genomic)",
            strat_style=LLM_STYLE,
        )
        fig_models_vs_perf_multiseed(
            aggc,
            outname="pi_llm_models_vs_perf_confirm_3seed",
            subtitle="Previous full-depth LLM runs (D=30k, genomic)",
        )
        written += [
            "pi_llm_round_progress_confirm",
            "pi_llm_models_vs_perf_confirm",
            "pi_llm_models_vs_perf_confirm_3seed",
        ]

    print("wrote:")
    for f in written:
        print(f"  {OUT}/{f}.png")


if __name__ == "__main__":
    main()
