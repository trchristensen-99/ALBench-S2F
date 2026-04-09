#!/usr/bin/env python
"""Add pretrained Malinois to the multi-model random bias comparison.

Loads the existing random bias NPZ, adds Malinois predictions, regenerates
the figure and summary table.

Usage:
    uv run --no-sync python scripts/add_malinois_random_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def one_hot_encode(seq: str) -> np.ndarray:
    """Encode DNA sequence to (4, L) one-hot array."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in mapping:
            arr[mapping[c], i] = 1.0
    return arr


def main():
    out_dir = Path("outputs/oracle_full_856k/random_bias_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing predictions
    existing = np.load(out_dir / "multi_model_random_preds.npz")
    results = {k: existing[k] for k in existing.files}
    logger.info("Loaded existing results: %s", list(results.keys()))

    # Generate same random sequences
    rng = np.random.default_rng(42)
    rand_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(10000)]

    # Load pretrained Malinois
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from scripts.eval_pretrained_malinois import load_pretrained_malinois

    model = load_pretrained_malinois("data/pretrained/malinois_trained/torch_checkpoint.pt", device)

    # Predict K562 (output index 0)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rand_seqs), 256):
            batch = rand_seqs[i : i + 256]
            encoded = np.stack([one_hot_encode(s) for s in batch])
            x = torch.from_numpy(encoded).float().to(device)
            out = model(x)[:, 0]  # K562 output
            x_rc = x.flip(-1)[:, [3, 2, 1, 0], :]
            out_rc = model(x_rc)[:, 0]
            avg = ((out + out_rc) / 2).cpu().numpy().reshape(-1)
            preds.append(avg)
    malinois_preds = np.concatenate(preds)
    results["Malinois_pretrained"] = malinois_preds
    logger.info("Malinois pretrained: mean=%.3f", np.mean(malinois_preds))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary table
    display_names = {
        "AG_S2_new_856K": "AG S2 (new, 856K)",
        "LegNet_Oracle": "LegNet Oracle",
        "DREAM-RNN_Oracle": "DREAM-RNN Oracle",
        "LegNet_real_labels": "LegNet (real labels)",
        "Malinois_pretrained": "Malinois (pretrained)",
    }

    print("\n" + "=" * 65)
    print("MULTI-MODEL RANDOM DNA PREDICTION COMPARISON (with Malinois)")
    print("=" * 65)
    print("\n%-30s %8s %8s %8s %8s" % ("Model", "Mean", "Median", "Std", "pct>0"))
    print("-" * 65)
    for key in sorted(results.keys()):
        p = results[key]
        name = display_names.get(key, key)
        print(
            "%-30s %8.3f %8.3f %8.3f %7.1f%%"
            % (name, np.mean(p), np.median(p), np.std(p), 100 * np.mean(p > 0))
        )

    # Save updated NPZ
    np.savez_compressed(out_dir / "multi_model_random_preds.npz", **results)

    # Generate figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.linspace(-3, 7, 80)
    colors = {
        "AG_S2_new_856K": "#1B5E20",
        "LegNet_Oracle": "#D4A017",
        "DREAM-RNN_Oracle": "#7B2D8E",
        "LegNet_real_labels": "#E8602C",
        "Malinois_pretrained": "#1565C0",
    }

    for key in sorted(results.keys()):
        p = results[key]
        name = display_names.get(key, key)
        c = colors.get(key, "#666666")
        ax.hist(
            p,
            bins=bins,
            alpha=0.35,
            color=c,
            label="%s (mean=%.2f)" % (name, np.mean(p)),
            density=True,
        )

    ax.axvline(0, color="k", ls="--", lw=1.5, label="Expected baseline (log2FC=0)")
    ax.set_xlabel("Predicted K562 log2FC for random DNA", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Multi-Model Random DNA Bias Comparison\n(10K random 200bp sequences)",
        fontsize=13,
    )
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        "results/peter_figures/multi_model_random_bias.png",
        dpi=200,
        bbox_inches="tight",
    )
    fig.savefig(
        "results/peter_figures/multi_model_random_bias.pdf",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print("\nSaved figure: results/peter_figures/multi_model_random_bias.png")


if __name__ == "__main__":
    main()
