#!/usr/bin/env python3
"""Model comparison bar plot in Alan Murphy's lentiMPRA style.

Matches the style from alphagenome_FT_MPRA:
- Grouped bars by test set (Reference, SNV Effect, Synthetic Seqs)
- Colors matching the lentiMPRA palette
- Value annotations rotated 90 degrees
- Clean seaborn-white style

Data loaded from benchmark_all_data.json which stores values for all 3 panels
(lentiMPRA, STARR-seq, episomal MPRA) for reproducibility.
"""

import json
from pathlib import Path

import matplotlib

# Font setup
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

available = set(f.name for f in fm.fontManager.ttflist)
for font in ["Calibri", "Arial", "Helvetica Neue", "Helvetica"]:
    if font in available:
        matplotlib.rcParams["font.family"] = font
        break

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
    }
)

DATA_PATH = (
    Path(__file__).parents[2] / "results" / "poster_stowers" / "data" / "benchmark_all_data.json"
)
all_data = json.load(open(DATA_PATH))

MODEL_COLORS = {
    "LegNet": "#E8DCCF",
    "MPRALegNet": "#E8DCCF",
    "DeepSTARR": "#E8DCCF",
    "DREAM-RNN": "#8B9DAF",
    "Dream-RNN": "#8B9DAF",
    "Malinois": "#B1934A",
    "Enf. (Probing)": "#E7CDC2",
    "Enf. (Fine-tuned)": "#A65141",
    "AG (Probing)": "#80A0C7",
    "AG (Fine-tuned)": "#394165",
}

# Dark enough colors for text annotations; light colors get a darker substitute
TEXT_COLORS = {
    "#E8DCCF": "#8B7D6B",
    "#E7CDC2": "#8B6B61",
    "#8B9DAF": "#5A6A7A",
}

out_dir = Path(__file__).parent / "model_comparison_plots"
out_dir.mkdir(exist_ok=True)


def make_barplot(
    ax, models_data, model_order, group_keys, title, ylim=(0.5, 1.0), group_display_labels=None
):
    """Create a single bar plot panel."""
    if group_display_labels is None:
        group_display_labels = group_keys
    n_groups = len(group_keys)
    n_models = len(model_order)
    width = 0.15
    x = np.arange(n_groups)

    for i, model in enumerate(model_order):
        v = models_data.get(model, {})
        color = MODEL_COLORS.get(model, "#888")

        means, stds = [], []
        for glabel in group_keys:
            vals = v.get(glabel)
            if vals is None:
                means.append(0)
                stds.append(0)
            elif isinstance(vals, list):
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if len(vals) > 1 else 0)
            else:
                means.append(float(vals))
                stds.append(0)

        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=[s if s > 0 else 0 for s in stds],
            capsize=3,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=1,
            alpha=0.9,
        )

        # Value annotations — centered on bar, above whisker
        txt_color = TEXT_COLORS.get(color, color)
        for bar_rect, val, err in zip(bars, means, stds):
            if val > 0:
                top = val + err  # position above whisker
                # Use bar center x from the rectangle directly
                cx = bar_rect.get_x() + bar_rect.get_width() / 2.0
                ax.annotate(
                    f"{val:.3f}",
                    xy=(cx, top),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    rotation=90,
                    color=txt_color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(group_display_labels)
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=14)
    ax.yaxis.grid(alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=9,
        ncol=2,
    )


# ── Episomal MPRA (our data) ──────────────────────────────────────────
# Data is nested by cell type — aggregate across cells (weighted equally)
ep_raw = all_data["episomal_MPRA"]
ep = {}
for model in [
    "LegNet",
    "DREAM-RNN",
    "Malinois",
    "Enf. (Probing)",
    "Enf. (Fine-tuned)",
    "AG (Probing)",
    "AG (Fine-tuned)",
]:
    m_data = ep_raw.get(model, {})
    if isinstance(m_data, str) or not m_data:
        continue
    all_ref, all_snv, all_syn = [], [], []
    for cell in ["k562", "hepg2", "sknsh"]:
        cv = m_data.get(cell, {})
        if isinstance(cv, dict) and "Reference" in cv:
            all_ref.extend(cv.get("Reference", []))
            all_snv.extend(cv.get("SNV Effect", []))
            all_syn.extend(cv.get("Synthetic Seqs", []))
    # Also handle old flat format (direct Reference/SNV Effect/Synthetic Seqs keys)
    if not all_ref and "Reference" in m_data:
        all_ref = m_data.get("Reference", [])
        all_snv = m_data.get("SNV Effect", [])
        all_syn = m_data.get("Synthetic Seqs", [])
    if all_ref:
        ep[model] = {"Reference": all_ref, "SNV Effect": all_snv, "Synthetic Seqs": all_syn}

ep_model_order = [
    "LegNet",
    "DREAM-RNN",
    "Enf. (Probing)",
    "Enf. (Fine-tuned)",
    "AG (Probing)",
    "AG (Fine-tuned)",
]
ep_model_order = [m for m in ep_model_order if m in ep]
ep_groups = ["Reference", "SNV Effect", "Synthetic Seqs"]
ep_display = ["Genomic Reference\nSequences", "SNV Effects", "High-Activity\nDesigned Sequences"]

fig, ax = plt.subplots(figsize=(9, 6))
make_barplot(
    ax,
    ep,
    ep_model_order,
    ep_groups,
    "Episomal MPRA",
    ylim=(0, 1.0),
    group_display_labels=ep_display,
)
fig.tight_layout()
fig.savefig(out_dir / "episomal_mpra_barplot.png", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "episomal_mpra_barplot.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: episomal_mpra_barplot")


# ── Episomal MPRA per-cell (3 panels like lentiMPRA) ─────────────────
# Build per-cell data: {model: {cell_label: [values]}}
ep_percell = {}
test_sets_for_percell = [
    ("Reference", "Genomic Ref."),
    ("SNV Effect", "SNV Effects"),
    ("Synthetic Seqs", "High-Act. Designed"),
]

for test_key, test_display in test_sets_for_percell:
    fig, axes_row = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    cell_labels = {"k562": "K562", "hepg2": "HepG2", "sknsh": "SK-N-SH"}

    for ax, cell_code in zip(axes_row, ["k562", "hepg2", "sknsh"]):
        cell_data = {}
        for model in ep_model_order:
            m_raw = ep_raw.get(model, {})
            cv = m_raw.get(cell_code, {})
            if isinstance(cv, dict) and test_key in cv and cv[test_key]:
                cell_data[model] = {test_key: cv[test_key]}

        cell_models = [m for m in ep_model_order if m in cell_data]
        if cell_models:
            make_barplot(
                ax,
                cell_data,
                cell_models,
                [test_key],
                cell_labels[cell_code],
                ylim=(0, 1.0),
                group_display_labels=[test_display],
            )

    fig.suptitle(
        f"Episomal MPRA — {test_display} (per cell type)", fontsize=15, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fname = f"episomal_mpra_percell_{test_key.lower().replace(' ', '_')}"
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# Also make a combined per-cell plot (all test sets, grouped by cell)
ep_by_cell = {}
for model in ep_model_order:
    m_raw = ep_raw.get(model, {})
    for cell_code, cell_label in [("k562", "K562"), ("hepg2", "HepG2"), ("sknsh", "SK-N-SH")]:
        cv = m_raw.get(cell_code, {})
        if isinstance(cv, dict) and cv.get("Reference"):
            # Compute combined (avg of 3 test sets)
            ref = cv.get("Reference", [])
            snv = cv.get("SNV Effect", [])
            syn = cv.get("Synthetic Seqs", [])
            if ref:
                ep_by_cell.setdefault(model, {})[cell_label] = ref

fig, ax = plt.subplots(figsize=(12, 6))
cell_order = ["K562", "HepG2", "SK-N-SH"]
cell_models = [m for m in ep_model_order if m in ep_by_cell]
make_barplot(
    ax,
    ep_by_cell,
    cell_models,
    cell_order,
    "Episomal MPRA — Genomic Reference (per cell type)",
    ylim=(0.7, 1.0),
)
fig.tight_layout()
fig.savefig(out_dir / "episomal_mpra_percell_ref_combined.png", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "episomal_mpra_percell_ref_combined.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: episomal_mpra_percell_ref_combined")


# ── lentiMPRA (Alan's data) ──────────────────────────────────────────
lenti = all_data["lentiMPRA"]
lenti_model_order = [
    "MPRALegNet",
    "Enf. (Probing)",
    "Enf. (Fine-tuned)",
    "AG (Probing)",
    "AG (Fine-tuned)",
]
lenti_groups = ["HepG2", "K562", "WTC11"]

fig, ax = plt.subplots(figsize=(8, 6))
make_barplot(ax, lenti, lenti_model_order, lenti_groups, "lentiMPRA", ylim=(0.5, 1.0))
fig.tight_layout()
fig.savefig(out_dir / "lentimpra_barplot.png", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "lentimpra_barplot.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: lentimpra_barplot")


# ── STARR-seq (Alan's data) ──────────────────────────────────────────
starr = all_data["STARR-seq"]
starr_model_order = [
    "DeepSTARR",
    "Dream-RNN",
    "Enf. (Probing)",
    "Enf. (Fine-tuned)",
    "AG (Probing)",
    "AG (Fine-tuned)",
]
starr_groups = ["Developmental", "House-keeping"]

fig, ax = plt.subplots(figsize=(8, 6))
make_barplot(ax, starr, starr_model_order, starr_groups, "STARR-seq", ylim=(0, 1.0))
fig.tight_layout()
fig.savefig(out_dir / "starrseq_barplot.png", dpi=200, bbox_inches="tight")
fig.savefig(out_dir / "starrseq_barplot.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved: starrseq_barplot")

print(f"\nAll plots saved to {out_dir}")
