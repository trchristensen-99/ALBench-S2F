#!/usr/bin/env python3
"""Generate PI-spec bar plot matching Alan Murphy's lentiMPRA style.

Models: Malinois, DREAM-RNN, Enformer (Probing + Fine-tuned), AG (Probing + Fine-tuned)
Metrics: Reference, SNV Effect, Synthetic Seqs
Colors: exact match from Alan's repo
Chr-split ref-only, weighted average across cell types, variance bars.
Legend on the right.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "alan_style_plots"
OUT.mkdir(parents=True, exist_ok=True)

# ── Exact colors from Alan's repo ──────────────────────────────────────
MODEL_ORDER = [
    "Malinois",
    "DREAM-RNN",
    "Enf. (Probing)",
    "Enf. (Fine-tuned)",
    "AG (Probing)",
    "AG (Fine-tuned)",
]

MODEL_COLORS = {
    "Malinois": "#E8DCCF",
    "DREAM-RNN": "#8B9DAF",
    "Enf. (Probing)": "#E7CDC2",
    "Enf. (Fine-tuned)": "#A65141",
    "AG (Probing)": "#80A0C7",
    "AG (Fine-tuned)": "#394165",
}


# ── Result directories (chr_split ref-only) ────────────────────────────
def _get_dirs(model, cell):
    dirs = {
        ("Malinois", "k562"): ["outputs/chr_split/k562/malinois"],
        ("Malinois", "hepg2"): ["outputs/chr_split/hepg2/malinois"],
        ("Malinois", "sknsh"): ["outputs/chr_split/sknsh/malinois"],
        ("DREAM-RNN", "k562"): ["outputs/chr_split/k562/dream_rnn"],
        ("DREAM-RNN", "hepg2"): ["outputs/chr_split/hepg2/dream_rnn"],
        ("DREAM-RNN", "sknsh"): ["outputs/chr_split/sknsh/dream_rnn"],
        ("Enf. (Probing)", "k562"): ["outputs/chr_split/k562/enformer_s1"],
        ("Enf. (Probing)", "hepg2"): ["outputs/chr_split/hepg2/enformer_s1"],
        ("Enf. (Probing)", "sknsh"): ["outputs/chr_split/sknsh/enformer_s1"],
        ("Enf. (Fine-tuned)", "k562"): ["outputs/enformer_k562_stage2_final"],
        ("Enf. (Fine-tuned)", "hepg2"): ["outputs/enformer_hepg2_stage2"],
        ("Enf. (Fine-tuned)", "sknsh"): ["outputs/enformer_sknsh_stage2"],
        ("AG (Probing)", "k562"): ["outputs/chr_split/k562/ag_all_folds_s1"],
        ("AG (Probing)", "hepg2"): ["outputs/chr_split/hepg2/ag_all_folds_s1"],
        ("AG (Probing)", "sknsh"): ["outputs/chr_split/sknsh/ag_all_folds_s1"],
        ("AG (Fine-tuned)", "k562"): ["outputs/chr_split/k562/ag_all_folds_s2"],
        ("AG (Fine-tuned)", "hepg2"): ["outputs/chr_split/hepg2/ag_all_folds_s2"],
        ("AG (Fine-tuned)", "sknsh"): ["outputs/chr_split/sknsh/ag_all_folds_s2"],
    }
    return dirs.get((model, cell), [])


METRICS = [
    ("Reference", ["in_distribution", "in_dist"]),
    ("SNV Effect", ["snv_delta"]),
    ("Synthetic Seqs", ["ood"]),
]

CELLS = ["k562", "hepg2", "sknsh"]
# Approximate sample counts per cell for weighting
CELL_WEIGHTS = {"k562": 40718, "hepg2": 40718, "sknsh": 40718}


def collect():
    """Collect metrics: model -> metric_name -> {"mean", "std", "n", "values"}."""
    results = {}
    for model in MODEL_ORDER:
        results[model] = {}
        for metric_name, key_variants in METRICS:
            # OOD only has K562 labels
            cells = ["k562"] if metric_name == "Synthetic Seqs" else CELLS

            cell_means = []
            cell_weights = []
            all_values = []

            for cell in cells:
                vals = []
                for d in _get_dirs(model, cell):
                    p = REPO / d
                    if not p.exists():
                        continue
                    for f in p.rglob("result.json"):
                        try:
                            data = json.loads(f.read_text())
                            tm = data.get("test_metrics", data)
                            for key in key_variants:
                                if key in tm and "pearson_r" in tm[key]:
                                    vals.append(tm[key]["pearson_r"])
                                    break
                        except Exception:
                            continue
                    if vals:
                        break

                if vals:
                    cell_means.append(np.mean(vals))
                    cell_weights.append(CELL_WEIGHTS[cell])
                    all_values.extend(vals)

            if cell_means:
                w = np.array(cell_weights, dtype=float)
                w /= w.sum()
                weighted_mean = float(np.average(cell_means, weights=w))
                std = float(np.std(all_values)) if len(all_values) > 1 else 0.0
                results[model][metric_name] = {
                    "mean": weighted_mean,
                    "std": std,
                    "n": len(all_values),
                    "values": all_values,
                }
            else:
                results[model][metric_name] = {"mean": 0, "std": 0, "n": 0, "values": []}

    return results


def plot(results):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    metric_names = [m[0] for m in METRICS]
    x_pos = np.arange(len(metric_names))
    n_models = len(MODEL_ORDER)
    width = 0.13

    for i, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        for mn in metric_names:
            m = results[model].get(mn, {"mean": 0, "std": 0})
            means.append(m["mean"])
            stds.append(m["std"])

        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(
            x_pos + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
        )

        # Value labels: italic, color-matched
        for bar, val, std_val in zip(bars, means, stds):
            if val > 0.01:
                y_top = val + std_val + 0.008
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_top,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontstyle="italic",
                    color=MODEL_COLORS[model],
                )

    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.set_title("lentiMPRA", fontsize=13, fontweight="semibold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim([0, 1.05])

    # Legend on the RIGHT (per PI request)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=9,
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(OUT / "mpra_benchmark_pi.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "mpra_benchmark_pi.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT / 'mpra_benchmark_pi.png'}")


def main():
    print("Collecting chr_split metrics for PI plot...")
    results = collect()

    # Print summary
    for model in MODEL_ORDER:
        for mn, _ in METRICS:
            m = results[model].get(mn, {})
            print(
                f"  {model:22s} {mn:15s}: "
                f"{m.get('mean', 0):.3f} ± {m.get('std', 0):.3f} "
                f"(n={m.get('n', 0)}, vals={[round(v, 4) for v in m.get('values', [])]})"
            )

    plot(results)


if __name__ == "__main__":
    main()
