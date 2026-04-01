#!/usr/bin/env python3
"""Plot Pareto frontier of in-dist vs OOD for regularization sweep.

Reads results from outputs/regularization_sweep/{model}/{config}/result.json
and plots in_dist Pearson R vs OOD Pearson R for each model.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "regularization_sweep"
OUT = REPO / "results" / "regularization_sweep"


def load_sweep_results(model: str) -> list[dict]:
    model_dir = BASE / model
    if not model_dir.exists():
        return []

    results = []
    for config_dir in sorted(model_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        for rj in config_dir.rglob("result.json"):
            r = json.loads(rj.read_text())
            tm = r.get("test_metrics", {})
            id_r = tm.get("in_distribution", tm.get("in_dist", {})).get("pearson_r")
            ood_r = tm.get("ood", {}).get("pearson_r")
            snv_a = tm.get("snv_abs", {}).get("pearson_r")
            snv_d = tm.get("snv_delta", {}).get("pearson_r")
            if id_r is not None:
                results.append(
                    {
                        "config": config_dir.name,
                        "in_dist": id_r,
                        "ood": ood_r if ood_r is not None else 0,
                        "snv_abs": snv_a if snv_a is not None else 0,
                        "snv_delta": snv_d if snv_d is not None else 0,
                    }
                )
    return results


def plot_pareto(model: str, results: list[dict], ax: plt.Axes):
    ids = [r["in_dist"] for r in results]
    oods = [r["ood"] for r in results]
    configs = [r["config"] for r in results]

    ax.scatter(ids, oods, s=40, alpha=0.7, zorder=3)

    # Label Pareto-optimal points
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, r2 in enumerate(results):
            if i != j and r2["in_dist"] >= r["in_dist"] and r2["ood"] >= r["ood"]:
                if r2["in_dist"] > r["in_dist"] or r2["ood"] > r["ood"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(i)

    for i in pareto:
        ax.scatter(ids[i], oods[i], s=80, c="red", marker="*", zorder=4)
        ax.annotate(
            configs[i],
            (ids[i], oods[i]),
            fontsize=5,
            rotation=30,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("In-distribution Pearson R")
    ax.set_ylabel("OOD Pearson R")
    ax.set_title(model, fontsize=13, fontweight="semibold")
    ax.grid(True, alpha=0.3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    models = ["enformer", "borzoi", "ntv3_post"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, models):
        results = load_sweep_results(model)
        if results:
            plot_pareto(model, results, ax)
            print(f"{model}: {len(results)} configs")
        else:
            ax.set_title(f"{model} (no data)")
            print(f"{model}: no results")

    fig.suptitle(
        "Regularization Pareto Frontier: In-dist vs OOD",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    for ext in (".png", ".pdf"):
        path = OUT / f"pareto_id_vs_ood{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
