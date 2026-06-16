"""Phase-0 (Exp0) summary figures: one simple, large-font figure per result.

Reads real data:
  - 0a SCREEN  : outputs/analysis_figures/llm_screen/screen_ranking.json
  - 0b ABLATION: outputs/hp_llm_ablation_e100/k562_genomic_d30000  (seed42_0 grid + seed42_1 _rep)
  - FOLLOWUP   : outputs/hp_llm_ablation_followup_e100/k562_genomic_d30000  (4 variants x 3 seeds)
  - ROUNDS     : cumulative-per-round curve from the followup n2 cells

Saves PNGs to outputs/analysis_figures/exp0_summary/. CPU/BLAS-bound -> srun cpuq, capped threads.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from albench.ensemble import fit_elasticnet_stack

plt.rcParams.update(
    {
        "font.size": 17,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "figure.constrained_layout.use": True,
    }
)

ABL = Path("outputs/hp_llm_ablation_e100/k562_genomic_d30000")
FU = Path("outputs/hp_llm_ablation_followup_e100/k562_genomic_d30000")
SCREEN = Path("outputs/analysis_figures/llm_screen/screen_ranking.json")
OUT = Path("outputs/analysis_figures/exp0_summary")
OUT.mkdir(parents=True, exist_ok=True)
NOISE = None  # filled after we compute baseline vs _rep


def ens_oracle(cell: Path):
    lab = np.load(cell / "labels.npz")
    vy, orc = lab["val_labels"], lab["test_oracle"]
    V, T = [], []
    for m in cell.glob("*_meta.json"):
        z = np.load(cell / m.name.replace("_meta.json", ".npz"))
        V.append(z["val_pred"])
        T.append(z["test_pred"])
    _, tp = fit_elasticnet_stack(np.vstack(V), vy, np.vstack(T))
    mt = np.isfinite(tp) & np.isfinite(orc)
    return float(pearsonr(tp[mt], orc[mt])[0])


def round_curve(cell: Path):
    lab = np.load(cell / "labels.npz")
    vy, orc = lab["val_labels"], lab["test_oracle"]
    byr = defaultdict(list)
    for m in cell.glob("*_meta.json"):
        d = json.load(open(m))
        z = np.load(cell / m.name.replace("_meta.json", ".npz"))
        byr[int(d["round"])].append((z["val_pred"], z["test_pred"]))
    V, T, c = [], [], []
    for r in sorted(byr):
        for vp, tp in byr[r]:
            V.append(vp)
            T.append(tp)
        _, pr = fit_elasticnet_stack(np.vstack(V), vy, np.vstack(T))
        mt = np.isfinite(pr) & np.isfinite(orc)
        c.append(float(pearsonr(pr[mt], orc[mt])[0]))
    return c


def fig_screen():
    d = json.load(open(SCREEN))
    rank = [r for r in d["ranking"] if r.get("complete")]
    rank = sorted(rank, key=lambda r: r["oracle_at_budget"], reverse=True)[:10][::-1]
    style_color = {
        "exploit": "#d62728",
        "critic": "#1f77b4",
        "explore": "#2ca02c",
        "neutral": "#7f7f7f",
        "default": "#9467bd",
    }
    names = ["{}  ({}, nv{})".format(r["style"], r["model"], r["novel"]) for r in rank]
    vals = [r["oracle_at_budget"] for r in rank]
    cols = [style_color.get(r["style"], "#333") for r in rank]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, "{:.3f}".format(v), va="center", fontsize=13)
    ax.set_xlim(min(vals) - 0.02, max(vals) + 0.012)
    ax.set_xlabel("ensemble oracle-r @ common GPU budget")
    ax.set_title(
        "Phase-0a Screen: best LLM proposer configs\n(exploit/critic personas + novel axes win; Sonnet ≥ Opus)"
    )
    present = ["exploit", "critic", "explore", "default", "neutral"]
    present = [s for s in present if any(r["style"] == s for r in rank)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=style_color[s]) for s in present]
    ax.legend(handles, present, title="persona", loc="lower right")
    fig.savefig(OUT / "01_screen_winners.png", dpi=140)
    plt.close(fig)


def _delta_bars(labels, title, fname, baseline_lab="llm_default_ctxnone"):
    base = ens_oracle(ABL / "seed42_0" / baseline_lab)
    deltas, names = [], []
    for lab in labels:
        cell = ABL / "seed42_0" / lab
        if not (cell / "labels.npz").exists():
            continue
        deltas.append(ens_oracle(cell) - base)
        names.append(lab.replace("llm_", "").replace("default_ctxnone_", ""))
    order = np.argsort(deltas)
    deltas = np.array(deltas)[order]
    names = [names[i] for i in order]
    cols = ["#2ca02c" if x > 2 * NOISE else "#d62728" if x < -2 * NOISE else "#999" for x in deltas]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(names))
    ax.barh(y, deltas, color=cols)
    ax.axvspan(-2 * NOISE, 2 * NOISE, color="gray", alpha=0.2, label="±2× noise floor")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Δ ensemble oracle-r  vs  ctxnone baseline")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def fig_fairness():
    labs = [
        "llm_default_ctxnone",
        "llm_default_ctxnokb",
        "llm_default_ctxfull",
        "llm_exploit_ctxnone",
        "llm_exploit_ctxnokb",
        "llm_exploit_ctxfull",
        "llm_blank_ctxnone",
        "llm_misguided_ctxnone",
    ]
    _delta_bars(
        labs,
        "Phase-0b Fairness: the LLM edge is the CONTEXT/KB handout,\nnot the persona (ctxfull lifts; persona at ctxnone is noise)",
        "02_fairness_context.png",
    )


def fig_mechanism():
    labs = [
        "llm_default_ctxnone_n1",
        "llm_default_ctxnone_n5",
        "llm_default_ctxnone_shuffle",
        "llm_default_ctxnone_nohist",
        "llm_default_ctxnone_hist5",
        "llm_default_ctxnone_histfull",
        "llm_default_ctxnone_chrono",
        "llm_default_ctxnone_worst",
    ]
    _delta_bars(
        labs,
        "Phase-0b Mechanism: proposals-per-call is the top lever\n(n1 worst); corrupting feedback hurts only modestly",
        "03_mechanism_feedback.png",
    )


def fig_followup():
    variants = [
        "llm_default_ctxnone",
        "llm_default_ctxnone_n8",
        "llm_diverse_ctxnone_n2",
        "llm_diverse_ctxnone_n8",
    ]
    seeds = ["seed42_0", "seed43_1", "seed44_2"]
    means, sds, names = [], [], []
    for v in variants:
        vals = [ens_oracle(FU / s / v) for s in seeds if (FU / s / v / "labels.npz").exists()]
        means.append(np.mean(vals))
        sds.append(np.std(vals))
        names.append(v.replace("llm_", "").replace("ctxnone", "ctx0"))
    base = means[0]
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(names))
    cols = ["#7f7f7f", "#d62728", "#2ca02c", "#2ca02c"]
    ax.bar(x, means, yerr=sds, capsize=8, color=cols)
    ax.axhspan(
        base - 2 * NOISE, base + 2 * NOISE, color="gray", alpha=0.2, label="baseline ±2× noise"
    )
    ax.axhline(base, color="k", lw=1, ls="--")
    for i, (mn, sd) in enumerate(zip(means, sds)):
        ax.text(
            i,
            mn + sd + 0.001,
            "{:+.3f}".format(mn - base) if i else "base",
            ha="center",
            fontsize=14,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0.70, max(means) + max(sds) + 0.01)
    ax.set_ylabel("ensemble oracle-r  (mean ± SD, 3 seeds)")
    ax.set_title(
        "Phase-0 Follow-up: 8 proposals/call gives NO lift;\nthe DIVERSE (decorrelation) proposer is the validated win"
    )
    ax.legend(loc="upper left")
    fig.savefig(OUT / "04_followup_proposer.png", dpi=140)
    plt.close(fig)


def fig_rounds():
    seeds = ["seed42_0", "seed43_1", "seed44_2"]
    variants = ["llm_default_ctxnone", "llm_diverse_ctxnone_n2"]
    pv = {}
    for v in variants:
        cs = [round_curve(FU / s / v) for s in seeds if (FU / s / v / ".ablation_done").exists()]
        L = min(len(c) for c in cs)
        pv[v] = np.array([c[:L] for c in cs])
    L = min(pv[v].shape[1] for v in variants)
    ov = np.mean([pv[v].mean(0)[:L] for v in variants], axis=0)
    rounds = np.arange(1, L + 1)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(rounds, pv["llm_default_ctxnone"].mean(0)[:L], "o-", lw=2, label="default proposer")
    ax.plot(rounds, pv["llm_diverse_ctxnone_n2"].mean(0)[:L], "o-", lw=2, label="diverse proposer")
    ax.plot(rounds, ov, "s-", color="k", lw=3, label="overall mean")
    ax.set_xlabel("AutoResearch rounds completed")
    ax.set_ylabel("cumulative ensemble oracle-r")
    ax.set_title(
        "AutoResearch improves monotonically with rounds\n(consistent same-sign gain every round → keep full 12 rounds)"
    )
    ax.set_xticks(rounds)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "05_rounds_curve.png", dpi=140)
    plt.close(fig)


def main():
    global NOISE
    base = ens_oracle(ABL / "seed42_0" / "llm_default_ctxnone")
    rep = ens_oracle(ABL / "seed42_1" / "llm_default_ctxnone_rep")
    NOISE = abs(base - rep)
    print("baseline={:.4f} rep={:.4f} noise={:.4f} (2x={:.4f})".format(base, rep, NOISE, 2 * NOISE))
    fig_screen()
    fig_fairness()
    fig_mechanism()
    fig_followup()
    fig_rounds()
    for p in sorted(OUT.glob("*.png")):
        print("saved", p)


if __name__ == "__main__":
    main()
