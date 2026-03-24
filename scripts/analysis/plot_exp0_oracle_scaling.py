#!/usr/bin/env python3
"""Plot Exp 0 oracle-label scaling curves."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "exp0_oracle_scaling_v4"
OUT = REPO / "results" / "exp0_oracle_scaling_v4"

POOL_SIZES = {"k562": 319742, "yeast": 6065324}
COLORS = {
    "alphagenome_k562_s1": "#1f77b4",
    "alphagenome_yeast_s1": "#1f77b4",
    "alphagenome_yeast_s2": "#9467bd",
    "dream_cnn": "#ff7f0e",
    "dream_rnn": "#2ca02c",
}
LABELS = {
    "alphagenome_k562_s1": "AG S1",
    "alphagenome_yeast_s1": "AG S1",
    "alphagenome_yeast_s2": "AG S2",
    "dream_cnn": "DREAM-CNN",
    "dream_rnn": "DREAM-RNN",
}


def load_results(task: str):
    raw: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    task_dir = BASE / task
    if not task_dir.exists():
        return {}
    for rj in task_dir.rglob("result.json"):
        r = json.loads(rj.read_text())
        parts = str(rj.relative_to(task_dir)).split("/")
        student = parts[0]
        n = r["n_train"]
        val = r.get("val_pearson_r", 0)
        hp = json.dumps(r["hp_config"], sort_keys=True)
        raw[(student, n)][hp].append(val)

    curves: dict[str, dict] = defaultdict(lambda: {"fracs": [], "vals": [], "stds": []})
    pool = POOL_SIZES[task]
    for student, n in sorted(raw.keys()):
        hp_scores = raw[(student, n)]
        best_hp = max(hp_scores.keys(), key=lambda k: np.mean(hp_scores[k]))
        vals = hp_scores[best_hp]
        frac = n / pool * 100
        curves[student]["fracs"].append(frac)
        curves[student]["vals"].append(np.mean(vals))
        curves[student]["stds"].append(np.std(vals) if len(vals) > 1 else 0)
    return curves


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (task, title) in enumerate([("k562", "K562"), ("yeast", "Yeast")]):
        ax = axes[idx]
        curves = load_results(task)

        for student in sorted(curves.keys()):
            d = curves[student]
            order = np.argsort(d["fracs"])
            fracs = [d["fracs"][i] for i in order]
            vals = [d["vals"][i] for i in order]
            stds = [d["stds"][i] for i in order]
            color = COLORS.get(student, None)
            label = LABELS.get(student, student)
            ax.errorbar(
                fracs,
                vals,
                yerr=stds,
                fmt="o-",
                label=label,
                color=color,
                markersize=6,
                linewidth=2,
                capsize=3,
            )

        ax.set_xlabel("Training Data Fraction (%)")
        ax.set_ylabel("Val Pearson R (oracle labels)")
        ax.set_title(f"{title} — Oracle Label Scaling (Exp 0)")
        ax.legend()
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if task == "k562":
            ax.set_ylim(0.3, 1.02)

    fig.tight_layout()
    out_path = OUT / "scaling_curves_preliminary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
