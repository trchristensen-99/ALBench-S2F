#!/usr/bin/env python3
"""Hyperparameter sensitivity analysis across data scales.

For each student model, plots one panel showing how each HP config
performs as training size increases. Highlights the best HP config
at each scale, revealing whether optimal HPs change with data scale.

Also produces a summary panel showing the best HP index at each size.

Data sources:
- outputs/exp0_oracle_scaling_v4/k562/{student}/random/n{size}/hp{idx}/seed{seed}/result.json
- outputs/exp1_1/k562/{student}/random/n{size}/hp{idx}/seed{seed}/result.json

Usage:
    python scripts/analysis/plot_hp_sensitivity.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "hp_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────

MODELS = {
    "LegNet": "legnet",
    "DREAM-CNN": "dream_cnn",
    "DREAM-RNN": "dream_rnn",
    "AG S1 (Probing)": "alphagenome_k562_s1",
    "AG S2 (Fine-tune)": "alphagenome_k562_s2",
}

# Colors for HP configs (up to 8)
HP_COLORS = [
    "#2176AE",  # blue
    "#E8563A",  # red-orange
    "#57A773",  # green
    "#9B59B6",  # purple
    "#D4A017",  # gold
    "#00BCD4",  # teal
    "#FF7043",  # deep orange
    "#8D6E63",  # brown
]


# ── Data loading ──────────────────────────────────────────────────────────


def _extract_pearson(result: dict) -> float | None:
    """Extract test in_dist Pearson R from a result dict."""
    tm = result.get("test_metrics", {})
    # Try both key conventions
    for key in ("in_dist", "in_distribution"):
        block = tm.get(key, {})
        if isinstance(block, dict) and "pearson_r" in block:
            return block["pearson_r"]
    return None


def load_hp_data(student_dir: str) -> dict:
    """Load all HP results for a student model.

    Returns:
        {size: {hp_idx: {"seeds": [pearson_values], "hp_config": dict}}}
    """
    data: dict[int, dict[int, dict]] = defaultdict(lambda: defaultdict(dict))

    # Search both experiment directories
    exp_dirs = [
        REPO / "outputs" / "exp0_oracle_scaling_v4" / "k562" / student_dir / "random",
        REPO / "outputs" / "exp1_1" / "k562" / student_dir / "random",
    ]

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        for size_dir in sorted(exp_dir.iterdir()):
            if not size_dir.is_dir() or not size_dir.name.startswith("n"):
                continue
            try:
                n = int(size_dir.name[1:])
            except ValueError:
                continue

            for hp_dir in sorted(size_dir.iterdir()):
                if not hp_dir.is_dir() or not hp_dir.name.startswith("hp"):
                    continue
                try:
                    hp_idx = int(hp_dir.name[2:])
                except ValueError:
                    continue

                seeds = []
                hp_config = {}
                for seed_dir in sorted(hp_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                        continue
                    rpath = seed_dir / "result.json"
                    if not rpath.exists():
                        continue
                    try:
                        result = json.loads(rpath.read_text())
                    except (json.JSONDecodeError, OSError):
                        continue
                    p = _extract_pearson(result)
                    if p is not None and not np.isnan(p):
                        seeds.append(p)
                    if not hp_config:
                        hp_config = result.get("hp_config", {})

                if seeds:
                    if n not in data or hp_idx not in data[n]:
                        data[n][hp_idx] = {"seeds": seeds, "hp_config": hp_config}
                    else:
                        # Merge seeds from both experiments (avoid duplicates)
                        existing = set(round(v, 8) for v in data[n][hp_idx]["seeds"])
                        for v in seeds:
                            if round(v, 8) not in existing:
                                data[n][hp_idx]["seeds"].append(v)

    return dict(data)


def _hp_label(hp_idx: int, hp_config: dict) -> str:
    """Format a short label for an HP config."""
    parts = [f"hp{hp_idx}"]
    if "learning_rate" in hp_config:
        lr = hp_config["learning_rate"]
        parts.append(f"lr={lr}")
    if "batch_size" in hp_config:
        bs = hp_config["batch_size"]
        parts.append(f"bs={bs}")
    return ", ".join(parts)


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_model_panel(ax, model_name: str, data: dict, show_ylabel: bool = True):
    """Plot HP sensitivity for one model on a given axes."""
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        return {}

    sizes = sorted(data.keys())
    all_hp_indices = sorted(set(hp for size_data in data.values() for hp in size_data))

    # Compute mean pearson per HP per size to find best
    hp_means: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for n in sizes:
        for hp_idx in all_hp_indices:
            if hp_idx in data[n]:
                mean_val = np.mean(data[n][hp_idx]["seeds"])
                hp_means[hp_idx].append((n, mean_val))

    # Find best HP at each size
    best_hp_per_size = {}
    for n in sizes:
        best_hp = None
        best_val = -np.inf
        for hp_idx in all_hp_indices:
            if hp_idx in data[n]:
                mean_val = np.mean(data[n][hp_idx]["seeds"])
                if mean_val > best_val:
                    best_val = mean_val
                    best_hp = hp_idx
        if best_hp is not None:
            best_hp_per_size[n] = best_hp

    # Determine which HP is best overall (most sizes where it wins)
    from collections import Counter

    win_counts = Counter(best_hp_per_size.values())
    overall_best_hp = win_counts.most_common(1)[0][0] if win_counts else None

    # Plot each HP config
    for hp_idx in all_hp_indices:
        color = HP_COLORS[hp_idx % len(HP_COLORS)]
        hp_sizes = []
        hp_means_vals = []
        hp_stds = []

        # Get label from any available config
        hp_config = {}
        for n in sizes:
            if hp_idx in data[n]:
                hp_config = data[n][hp_idx].get("hp_config", {})
                if hp_config:
                    break

        label = _hp_label(hp_idx, hp_config)
        is_best = hp_idx == overall_best_hp

        for n in sizes:
            if hp_idx in data[n]:
                vals = data[n][hp_idx]["seeds"]
                hp_sizes.append(n)
                hp_means_vals.append(np.mean(vals))
                hp_stds.append(np.std(vals))

                # Plot individual seed points
                for v in vals:
                    ax.scatter(
                        n,
                        v,
                        color=color,
                        s=12,
                        alpha=0.3,
                        zorder=2,
                        edgecolors="none",
                    )

        if hp_sizes:
            lw = 2.5 if is_best else 1.2
            alpha = 1.0 if is_best else 0.6
            marker = "o" if is_best else "s"
            ms = 6 if is_best else 4

            ax.plot(
                hp_sizes,
                hp_means_vals,
                color=color,
                linewidth=lw,
                alpha=alpha,
                marker=marker,
                markersize=ms,
                label=label + (" *" if is_best else ""),
                zorder=3 if is_best else 2,
            )

            # Shade std
            lower = np.array(hp_means_vals) - np.array(hp_stds)
            upper = np.array(hp_means_vals) + np.array(hp_stds)
            ax.fill_between(
                hp_sizes,
                lower,
                upper,
                color=color,
                alpha=0.08 if not is_best else 0.15,
                zorder=1,
            )

    ax.set_xscale("log")
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training set size", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Test in-dist Pearson R", fontsize=10)

    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Legend
    leg = ax.legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor="none")
    leg.get_frame().set_facecolor("white")

    return best_hp_per_size


def plot_summary_panel(ax, all_best_hps: dict[str, dict[int, int]]):
    """Summary heatmap: best HP index at each training size for each model."""
    # Collect all sizes
    all_sizes = sorted(set(n for bests in all_best_hps.values() for n in bests.keys()))
    model_names = list(all_best_hps.keys())

    if not all_sizes or not model_names:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Build matrix
    matrix = np.full((len(model_names), len(all_sizes)), np.nan)
    for i, model in enumerate(model_names):
        for j, n in enumerate(all_sizes):
            if n in all_best_hps[model]:
                matrix[i, j] = all_best_hps[model][n]

    ax.imshow(matrix, aspect="auto", cmap="Set2", vmin=-0.5, vmax=3.5, interpolation="none")

    # Labels
    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels([f"n={n:,}" for n in all_sizes], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(all_sizes)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"hp{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                )

    ax.set_title("Best HP config by model and data scale", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("Loading HP sensitivity data...")

    all_data = {}
    all_best_hps = {}

    for display_name, student_dir in MODELS.items():
        print(f"  {display_name} ({student_dir})...")
        data = load_hp_data(student_dir)
        all_data[display_name] = data

        # Print summary
        sizes = sorted(data.keys())
        for n in sizes:
            hp_indices = sorted(data[n].keys())
            parts = []
            for hp_idx in hp_indices:
                vals = data[n][hp_idx]["seeds"]
                parts.append(
                    f"hp{hp_idx}: {np.mean(vals):.4f} +/- {np.std(vals):.4f} ({len(vals)} seeds)"
                )
            print(f"    n={n:>7d}: {', '.join(parts)}")

    # ── Figure 1: Multi-panel per-model HP sensitivity ────────────────────

    n_models = len(MODELS)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for idx, (display_name, student_dir) in enumerate(MODELS.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        best_hps = plot_model_panel(
            ax, display_name, all_data[display_name], show_ylabel=(col == 0)
        )
        all_best_hps[display_name] = best_hps

    # Hide empty subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Hyperparameter sensitivity across data scales (K562)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    path1 = OUT / "hp_sensitivity_per_model.pdf"
    fig.savefig(path1, bbox_inches="tight", dpi=200)
    fig.savefig(path1.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"\nSaved: {path1}")
    plt.close(fig)

    # ── Figure 2: Summary heatmap ─────────────────────────────────────────

    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    plot_summary_panel(ax2, all_best_hps)
    fig2.tight_layout()
    path2 = OUT / "hp_best_config_summary.pdf"
    fig2.savefig(path2, bbox_inches="tight", dpi=200)
    fig2.savefig(path2.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {path2}")
    plt.close(fig2)

    # ── Figure 3: Combined (all models overlaid) ──────────────────────────
    # One panel per HP config, showing all models — alternative view

    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    model_colors = {
        "LegNet": "#D4A017",
        "DREAM-CNN": "#9B59B6",
        "DREAM-RNN": "#8B9DAF",
        "AG S1 (Probing)": "#2176AE",
        "AG S2 (Fine-tune)": "#E8563A",
    }

    for display_name, data in all_data.items():
        if not data:
            continue
        sizes = sorted(data.keys())
        color = model_colors.get(display_name, "#333333")

        # For each size, plot mean across ALL HPs (thin) and best HP (bold)
        best_means = []
        all_hp_means = []
        valid_sizes = []

        for n in sizes:
            hp_vals = []
            for hp_idx, hp_data in data[n].items():
                hp_vals.append(np.mean(hp_data["seeds"]))
            if hp_vals:
                valid_sizes.append(n)
                best_means.append(max(hp_vals))
                all_hp_means.append(hp_vals)

        if valid_sizes:
            # Plot spread: min to max across HPs
            mins = [min(v) for v in all_hp_means]
            maxs = [max(v) for v in all_hp_means]
            ax3.fill_between(
                valid_sizes,
                mins,
                maxs,
                color=color,
                alpha=0.15,
                zorder=1,
            )
            ax3.plot(
                valid_sizes,
                best_means,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=f"{display_name} (best HP)",
                zorder=3,
            )
            ax3.plot(
                valid_sizes,
                mins,
                color=color,
                linewidth=0.8,
                linestyle="--",
                alpha=0.5,
                zorder=2,
            )

    ax3.set_xscale("log")
    ax3.set_xlabel("Training set size", fontsize=11)
    ax3.set_ylabel("Test in-dist Pearson R", fontsize=11)
    ax3.set_title(
        "HP sensitivity range per model (best HP vs worst HP)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="none")
    fig3.tight_layout()
    path3 = OUT / "hp_sensitivity_range.pdf"
    fig3.savefig(path3, bbox_inches="tight", dpi=200)
    fig3.savefig(path3.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {path3}")
    plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
