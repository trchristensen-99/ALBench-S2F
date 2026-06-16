"""Strategy-selection figures for the Phase-0 LLM screen.

Explains, from the REAL screen_ranking.json, why the three deploy proposers were chosen
and how novel-axes affect each persona. Saves large-font PNGs to
outputs/analysis_figures/exp0_strategy_choice/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.constrained_layout.use": True,
    }
)

SCREEN = Path("outputs/analysis_figures/llm_screen/screen_ranking.json")
OUT = Path("outputs/analysis_figures/exp0_strategy_choice")
OUT.mkdir(parents=True, exist_ok=True)
NOISE = 0.008  # single-run replicate floor (ens oracle-r)

d = json.load(open(SCREEN))
R = [x for x in d["ranking"] if x.get("complete")]


def get(style, model, novel):
    for x in R:
        if x["style"] == style and x["model"] == model and str(x.get("novel")) == str(novel):
            return x["oracle_at_budget"]
    return None


STYLES = ["exploit", "critic", "diverse", "default", "neutral", "explore"]
CHOSEN = {"exploit": "1", "critic": "0", "diverse": "0"}  # current bundle picks (nv)


def fig_model_persona():
    """Grouped bars: oracle@B per persona, Sonnet vs Opus, at each persona's BEST novel."""
    son, opu, labs, picks = [], [], [], []
    for s in STYLES:
        sv = max([v for v in (get(s, "sonnet", "0"), get(s, "sonnet", "1")) if v is not None])
        ov = [v for v in (get(s, "opus", "0"), get(s, "opus", "1")) if v is not None]
        ov = max(ov) if ov else 0.0
        son.append(sv)
        opu.append(ov)
        labs.append(s)
        picks.append(s in CHOSEN)
    x = np.arange(len(STYLES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 7))
    b1 = ax.bar(x - w / 2, son, w, label="Sonnet", color="#1f77b4")
    ax.bar(x + w / 2, opu, w, label="Opus", color="#ff7f0e")
    for i, p in enumerate(picks):
        if p:
            ax.text(x[i], son[i] + 0.004, "★ chosen", ha="center", color="#d62728", fontsize=13)
            b1[i].set_edgecolor("#d62728")
            b1[i].set_linewidth(3)
    for i in range(len(STYLES)):
        ax.text(
            x[i] - w / 2,
            son[i] - 0.018,
            "{:.3f}".format(son[i]),
            ha="center",
            color="w",
            fontsize=12,
        )
    ax.axhline(max(son), color="#1f77b4", ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(STYLES, rotation=10)
    ax.set_ylabel("ensemble oracle-r @ GPU budget (best novel-axis)")
    ax.set_ylim(0.66, max(son) + 0.02)
    ax.set_title(
        "Phase-0a Screen: persona × model\nSonnet ≥ Opus everywhere; exploit / critic / diverse are the style-distinct contenders"
    )
    ax.legend(loc="upper right")
    fig.savefig(OUT / "A_model_persona.png", dpi=140)
    plt.close(fig)


def fig_novel_effect():
    """nv1 - nv0 delta on Sonnet per persona, with the noise band."""
    deltas, labs, cols = [], [], []
    for s in STYLES:
        a, b = get(s, "sonnet", "0"), get(s, "sonnet", "1")
        if a is None or b is None:
            continue
        dl = b - a
        deltas.append(dl)
        labs.append(s)
        cols.append("#2ca02c" if dl > NOISE else "#d62728" if dl < -NOISE else "#999")
    order = np.argsort(deltas)
    deltas = np.array(deltas)[order]
    labs = [labs[i] for i in order]
    cols = [cols[i] for i in order]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(labs))
    ax.barh(y, deltas, color=cols)
    ax.axvspan(-NOISE, NOISE, color="gray", alpha=0.2, label="±1× noise floor")
    ax.axvline(0, color="k", lw=1)
    for i, v in enumerate(deltas):
        off = 0.0006 if v >= 0 else 0.0006  # always sit just inside/right of the tip
        ax.text(v + off, i, "{:+.3f}".format(v), va="center", ha="left", fontsize=13)
    ax.set_xlim(min(deltas) - 0.004, max(deltas) + 0.004)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("Δ oracle-r  (novel-axes ON  −  OFF), Sonnet")
    ax.set_title(
        "Novel-axes help EXPLOIT, hurt CRITIC, ~neutral for DIVERSE\n→ each proposer deployed at its own best novel setting"
    )
    ax.legend(loc="lower right")
    fig.savefig(OUT / "B_novel_axis_effect.png", dpi=140)
    plt.close(fig)


def fig_why_three():
    """The 3 picks: screen oracle@B (at chosen novel) + the 3-seed followup diverse win."""
    # 3-seed followup ens-oracle means (computed earlier from the followup cells, n=24/cell):
    fu = {"default": 0.7188, "diverse_n2": 0.7353, "n8": 0.7255}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    # left: screen oracle@B for the 3 picks (sonnet, chosen novel) + best alternative dropped
    names = ["exploit nv1", "critic nv0", "diverse nv0", "explore nv0\n(dropped)"]
    vals = [
        get("exploit", "sonnet", "1"),
        get("critic", "sonnet", "0"),
        get("diverse", "sonnet", "0"),
        get("explore", "sonnet", "0"),
    ]
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#bbbbbb"]
    ax = axes[0]
    xb = np.arange(len(names))
    ax.bar(xb, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(xb[i], v + 0.002, "{:.3f}".format(v), ha="center", fontsize=14)
    ax.set_xticks(xb)
    ax.set_xticklabels(names)
    ax.set_ylim(0.66, max(vals) + 0.012)
    ax.set_ylabel("screen ensemble oracle-r @ budget")
    ax.set_title("Single-seed screen (Sonnet)\ntwo top personas + a 3rd style-distinct proposer")
    # right: 3-seed followup — the multi-seed validation that diverse is a real win
    fn = ["default\n(baseline)", "n8\n(wider batch)", "diverse n2\n(decorrelation)"]
    fv = [fu["default"], fu["n8"], fu["diverse_n2"]]
    base = fu["default"]
    ax = axes[1]
    xf = np.arange(len(fn))
    ax.bar(xf, fv, color=["#7f7f7f", "#d62728", "#2ca02c"])
    ax.axhspan(
        base - 2 * NOISE, base + 2 * NOISE, color="gray", alpha=0.2, label="baseline ±2× noise"
    )
    ax.axhline(base, color="k", ls="--", lw=1)
    for i, v in enumerate(fv):
        ax.text(
            xf[i],
            v + 0.001,
            ("base" if i == 0 else "{:+.3f}".format(v - base)),
            ha="center",
            fontsize=14,
        )
    ax.set_xticks(xf)
    ax.set_xticklabels(fn)
    ax.set_ylim(0.70, max(fv) + 0.008)
    ax.set_ylabel("ens oracle-r (3-seed mean)")
    ax.set_title("3-seed follow-up: diverse is the validated win\n(wider batch n8 gives no lift)")
    ax.legend(loc="upper left")
    fig.suptitle(
        "Why these three: two strongest personas (decorrelated) + the multi-seed-validated diversity proposer",
        fontsize=18,
    )
    fig.savefig(OUT / "C_why_these_three.png", dpi=140)
    plt.close(fig)


fig_model_persona()
fig_novel_effect()
fig_why_three()
for p in sorted(OUT.glob("*.png")):
    print("saved", p)
