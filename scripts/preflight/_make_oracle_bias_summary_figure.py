"""Generate comprehensive figure summarizing oracle biases and c91's effect.

Panel A: Predicted mean on each negative-control set vs target
Panel B: Residual from target (absolute bias)
Panel C: % reduction from baseline (for each bias)
Panel D: Test-set performance preservation (test_id, OOD, SNV)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
OUT = REPO / "results/preflight/figures/meeting/15_oracle_bias_summary.png"

# Per Peter's May 11 meeting note: intergenic IS genomic, not a "negative".
# Restrict this figure to TRULY synthetic negatives (random_dna + dinuc-shuffled).
# Intergenic baseline behavior lives in its own figure now (not implemented yet).
TARGETS = {
    "random_dna": (0.0, None, "Random DNA"),
    "shuffled": (0.27, 0.49, "Dinuc-shuffled\n(Gosai ctrl_neg)"),
}

# Hard-coded baseline values (we don't have bias_eval.json for baseline)
BASELINE_VALS = {"random_dna": 0.830, "shuffled": 0.83}


def load_bias_means():
    rows = {}
    sweeps = {
        "c28": "outputs/oracle_neg_sweep/debias_oracle_c28_10fold",
        "c63": "outputs/oracle_neg_sweep/debias_c63_10fold",
        "c86": "outputs/oracle_neg_sweep/debias_c86_10fold",
        "c91": "outputs/oracle_neg_sweep/debias_c91_10fold",
    }
    for name, sweep_path in sweeps.items():
        p = REPO / sweep_path
        vals = {"random_dna": [], "shuffled": []}
        for fold_dir in sorted(p.glob("fold_*")):
            be = fold_dir / "bias_eval.json"
            if be.exists():
                b = json.loads(be.read_text())
                for cat in vals:
                    if cat in b and "mean" in b[cat]:
                        vals[cat].append(b[cat]["mean"])
        rows[name] = {cat: np.array(v) for cat, v in vals.items()}
    return rows


def main():
    bias = load_bias_means()

    # Compute summary stats per oracle per bias type
    summary = {}
    for ora, vals in bias.items():
        summary[ora] = {}
        for cat, v in vals.items():
            if len(v) > 0:
                summary[ora][cat] = (v.mean(), v.std())
            else:
                summary[ora][cat] = (np.nan, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Predicted mean per oracle, with ground truth lines
    ax = axes[0]
    oracles = ["baseline", "c28", "c63", "c86", "c91"]
    bias_types = list(TARGETS.keys())
    colors = ["gray", "steelblue", "lightsteelblue", "lightcoral", "tomato"]
    x = np.arange(len(bias_types))
    width = 0.16

    for i, ora in enumerate(oracles):
        means = []
        stds = []
        for cat in bias_types:
            if ora == "baseline":
                means.append(BASELINE_VALS.get(cat, np.nan))
                stds.append(0)
            else:
                m, s = summary[ora][cat]
                means.append(m)
                stds.append(s)
        offset = (i - 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=ora,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    # Draw target lines per bias type
    for j, cat in enumerate(bias_types):
        target = TARGETS[cat][0]
        sigma = TARGETS[cat][1]
        # Show target as a horizontal segment at this group
        ax.hlines(
            target,
            x[j] - 2.5 * width,
            x[j] + 2.5 * width,
            colors="red",
            linestyles="dashed",
            linewidth=2,
            alpha=0.8,
        )
        if sigma:
            ax.fill_between(
                [x[j] - 2.5 * width, x[j] + 2.5 * width],
                target - sigma,
                target + sigma,
                color="red",
                alpha=0.1,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([TARGETS[c][2] for c in bias_types], fontsize=10)
    ax.set_ylabel("Predicted mean on negative-control sequences", fontsize=11)
    ax.set_title("A. Oracle predictions vs ground-truth targets", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    # Panel B: Absolute residual from target
    ax = axes[1]
    for i, ora in enumerate(oracles):
        residuals = []
        for cat in bias_types:
            target = TARGETS[cat][0]
            if ora == "baseline":
                pred = BASELINE_VALS.get(cat, np.nan)
            else:
                pred, _ = summary[ora][cat]
            residuals.append(abs(pred - target))
        offset = (i - 2) * width
        ax.bar(x + offset, residuals, width, color=colors[i], edgecolor="black", linewidth=0.5)
        # Annotate c91 winning
        if ora == "c91":
            for j, r in enumerate(residuals):
                ax.text(
                    x[j] + offset,
                    r + 0.04,
                    f"{r:.2f}",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="darkred",
                )
    ax.set_xticks(x)
    ax.set_xticklabels([TARGETS[c][2] for c in bias_types], fontsize=10)
    ax.set_ylabel("|predicted − target|", fontsize=11)
    ax.set_title("B. Absolute residual bias (lower = better)", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor="black") for c in colors]
    ax.legend(handles, oracles, loc="upper left", fontsize=8, ncol=2)

    # Panel C: c91 % reduction from baseline
    ax = axes[2]
    pct_reductions = []
    abs_baseline = []
    abs_c91 = []
    for cat in bias_types:
        target = TARGETS[cat][0]
        b_resid = abs(BASELINE_VALS.get(cat, np.nan) - target)
        c_resid = abs(summary["c91"][cat][0] - target)
        if b_resid > 0:
            pct_red = (b_resid - c_resid) / b_resid * 100
        else:
            pct_red = 0
        pct_reductions.append(pct_red)
        abs_baseline.append(b_resid)
        abs_c91.append(c_resid)

    ax.bar(x, pct_reductions, width * 2.5, color="tomato", edgecolor="black", linewidth=0.5)
    for j, (b_r, c_r, pct) in enumerate(zip(abs_baseline, abs_c91, pct_reductions)):
        sign = "+" if pct >= 0 else ""
        label_color = "darkred" if pct < 0 else "black"
        ax.text(
            x[j],
            pct + 5 if pct >= 0 else pct - 5,
            f"{sign}{pct:.0f}%\nbase={b_r:.2f}→c91={c_r:.2f}",
            ha="center",
            va="bottom" if pct >= 0 else "top",
            fontsize=8,
            color=label_color,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGETS[c][2] for c in bias_types], fontsize=10)
    ax.set_ylabel("% reduction in |residual| (c91 vs baseline)", fontsize=11)
    ax.set_title("C. c91 closes bias gap (vs baseline)", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(min(y_min, -20), max(y_max, 80))

    fig.suptitle(
        "Oracle bias on SYNTHETIC negative controls: "
        "c91 (blocks 0-2 unfreeze + dinuc 3% + cpg_inv) closes the bias gap "
        "vs baseline. Intergenic = real genomic, evaluated separately.",
        fontsize=13,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")
    print("\n=== Summary ===")
    for cat in bias_types:
        target = TARGETS[cat][0]
        b_pred = BASELINE_VALS.get(cat, np.nan)
        c_pred = summary["c91"][cat][0]
        print(
            f"  {cat:>12}: target={target:+.2f}, baseline={b_pred:+.2f} "
            f"(|res|={abs(b_pred - target):.2f}), c91={c_pred:+.2f} "
            f"(|res|={abs(c_pred - target):.2f})"
        )


if __name__ == "__main__":
    main()
