"""Activity (oracle label) distributions for reservoir-sampled training data + test sets.

Shows how the activity distribution of each reservoir sampling strategy compares
to each test set — gives intuition for which reservoirs cover which test-distribution
regions.

Two-panel layout:
  Left: training reservoirs (KDEs)
  Right: test sets (KDEs)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/activity_distributions"


def _palette(n):
    # 20 visually distinct colors from tab20
    cmap = cm.get_cmap("tab20")
    return [cmap(i / 20.0) for i in range(n)]


RES_DATA = [
    (
        "genomic",
        "Genomic",
        REPO / "outputs/chr_split_cache/chr_train_ref_only.npz",
        "oracle_labels",
    ),
    (
        "random",
        "Random",
        REPO / "outputs/reservoir_cache/k562_random_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "prm_1pct",
        "PRM 1%",
        REPO / "outputs/reservoir_cache/k562_prm_1pct_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "prm_5pct",
        "PRM 5%",
        REPO / "outputs/reservoir_cache/k562_prm_5pct_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "prm_10pct",
        "PRM 10%",
        REPO / "outputs/reservoir_cache/k562_prm_10pct_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "prm_20pct",
        "PRM 20%",
        REPO / "outputs/reservoir_cache/k562_prm_20pct_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "prm_attribution_1pct",
        "PRM Attribution 1%",
        REPO / "outputs/reservoir_cache/k562_prm_attribution_1pct_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "prm_uncertainty_1pct",
        "PRM Uncertainty 1%",
        REPO / "outputs/reservoir_cache/k562_prm_uncertainty_1pct_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "evoaug_heavy",
        "EvoAug Heavy",
        REPO / "outputs/reservoir_cache/k562_evoaug_heavy_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "evoaug_structural",
        "EvoAug Structural",
        REPO / "outputs/reservoir_cache/k562_evoaug_structural_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "motif_planted",
        "Motif Planted",
        REPO / "outputs/reservoir_cache/k562_motif_planted_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "motif_planted_v2",
        "Motif Planted v2 (Shuffled)",
        REPO / "outputs/reservoir_cache/k562_motif_planted_v2_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "motif_shuffled",
        "Motif Shuffled (original)",
        REPO / "outputs/reservoir_cache/k562_motif_shuffled_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "motif_grammar",
        "Motif Grammar",
        REPO / "outputs/reservoir_cache/k562_motif_grammar_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "phylogenetic_zoonomia",
        "Phylogenetic (Zoonomia)",
        REPO / "outputs/reservoir_cache/k562_phylogenetic_zoonomia_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "dinuc_shuffle",
        "Dinuc-shuffled",
        REPO / "outputs/reservoir_cache/k562_dinuc_shuffle_d1000000_seed42.npz",
        "oracle_labels",
    ),
    (
        "gc_matched",
        "GC-matched",
        REPO / "outputs/reservoir_cache/k562_gc_matched_d1000000_seed42.npz",
        "oracle_labels",
    ),
]
_res_colors = _palette(len(RES_DATA))
RESERVOIRS = [(k, lab, _res_colors[i], path, key) for i, (k, lab, path, key) in enumerate(RES_DATA)]

TEST_DIR = REPO / "data/k562/test_sets_ag_s2_chrsplit"
TST_DATA = [
    ("genomic", "Genomic (chr 7+13)", TEST_DIR / "genomic_oracle.npz", "oracle_mean"),
    ("ood", "High-Activity Designed", TEST_DIR / "ood_oracle.npz", "oracle_mean"),
    ("snv_ref", "SNV Ref", TEST_DIR / "snv_oracle.npz", "ref_mean"),
    ("snv_alt", "SNV Alt", TEST_DIR / "snv_oracle.npz", "alt_mean"),
    ("random_32k", "Random 32k", TEST_DIR / "random_32k_oracle.npz", "oracle_mean"),
    ("dinuc_shuffle", "Dinuc-shuffled", TEST_DIR / "dinuc_shuffle_oracle.npz", "oracle_mean"),
    ("sub_low", "Substitution (low)", TEST_DIR / "sub_low_oracle.npz", "oracle_mean"),
    ("sub_med", "Substitution (med)", TEST_DIR / "sub_med_oracle.npz", "oracle_mean"),
    ("sub_high", "Substitution (high)", TEST_DIR / "sub_high_oracle.npz", "oracle_mean"),
    ("ins_low", "Insertion (low)", TEST_DIR / "ins_low_oracle.npz", "oracle_mean"),
    ("ins_med", "Insertion (med)", TEST_DIR / "ins_med_oracle.npz", "oracle_mean"),
    ("ins_high", "Insertion (high)", TEST_DIR / "ins_high_oracle.npz", "oracle_mean"),
    ("del_low", "Deletion (low)", TEST_DIR / "del_low_oracle.npz", "oracle_mean"),
    ("del_med", "Deletion (med)", TEST_DIR / "del_med_oracle.npz", "oracle_mean"),
    ("del_high", "Deletion (high)", TEST_DIR / "del_high_oracle.npz", "oracle_mean"),
    ("translocation", "Translocation", TEST_DIR / "translocation_oracle.npz", "oracle_mean"),
    ("inversion", "Inversion", TEST_DIR / "inversion_oracle.npz", "oracle_mean"),
]
_tst_colors = _palette(len(TST_DATA))
TEST_SETS = [(k, lab, _tst_colors[i], path, key) for i, (k, lab, path, key) in enumerate(TST_DATA)]


def load_labels(path: Path, key: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if key in z.files:
        y = z[key].astype(np.float32)
    elif "oracle_labels" in z.files:
        y = z["oracle_labels"].astype(np.float32)
    elif "oracle_mean" in z.files:
        y = z["oracle_mean"].astype(np.float32)
    else:
        raise KeyError(f"no label key found in {path}, files={z.files}")
    return y[np.isfinite(y)]


def plot_dist(ax, items, title):
    bins = np.linspace(-3, 8, 80)
    for name, label, color, path, key in items:
        if not path.exists():
            print(f"  skip {name}: missing {path}")
            continue
        y = load_labels(path, key)
        hist, edges = np.histogram(y, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(
            centers,
            hist,
            color=color,
            label=f"{label}  (μ={y.mean():.2f}, σ={y.std():.2f}, n={len(y):,})",
            linewidth=1.8,
            alpha=0.85,
        )
    ax.set_xlabel("Oracle activity (log2FC)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92, ncol=1)
    ax.set_xlim(-3, 8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 2-panel side-by-side (combined)
    fig, axes = plt.subplots(1, 2, figsize=(22, 7))
    plot_dist(
        axes[0], RESERVOIRS, f"Training reservoirs ({len(RESERVOIRS)} strategies, D=1M sample)"
    )
    plot_dist(axes[1], TEST_SETS, f"Test sets ({len(TEST_SETS)} panels)")
    fig.tight_layout()
    fig.savefig(OUT / "activity_distributions_all.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "activity_distributions_all.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved combined 2-panel")

    # Standalone: reservoirs only
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_dist(
        ax, RESERVOIRS, f"All reservoir sampling strategies ({len(RESERVOIRS)} total, D=1M sample)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "activity_distributions_reservoirs.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "activity_distributions_reservoirs.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved reservoirs-only")

    # Standalone: test sets only
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_dist(ax, TEST_SETS, f"All test sets ({len(TEST_SETS)} panels)")
    fig.tight_layout()
    fig.savefig(OUT / "activity_distributions_test_sets.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "activity_distributions_test_sets.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved test-sets-only")

    # Also print a table of moments
    print("\n=== Activity moments (oracle log2FC) ===")
    print(f"  {'item':<28}  {'n':>10}  {'mean':>7}  {'std':>6}  {'q05':>7}  {'q50':>7}  {'q95':>7}")
    for items, kind in [(RESERVOIRS, "RES"), (TEST_SETS, "TST")]:
        for name, label, _, path, key in items:
            if not path.exists():
                continue
            y = load_labels(path, key)
            print(
                f"  [{kind}] {label:<22}  {len(y):>10,}  {y.mean():>7.3f}  {y.std():>6.3f}  "
                f"{np.percentile(y, 5):>7.3f}  {np.percentile(y, 50):>7.3f}  {np.percentile(y, 95):>7.3f}"
            )


if __name__ == "__main__":
    main()
