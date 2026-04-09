#!/usr/bin/env python
"""Predict activity for Agarwal et al. K562 controls + random DNA with multiple models.

Tests AG S2 oracle, Malinois (pretrained), DREAM-RNN oracle, LegNet oracle,
and LegNet (real labels) on:
  1. 250 dinucleotide-shuffled negative controls (Agarwal et al. 2025)
  2. 200 Ernst et al. 2016 negative controls
  3. 50 Ernst et al. 2016 positive controls
  4. 10K random 200bp DNA sequences

Generates comparison figure showing model predictions for each category.

Usage:
    uv run --no-sync python scripts/predict_controls_multimodel.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(".")


def one_hot_encode(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, c in enumerate(seq.upper()):
        if c in mapping:
            arr[mapping[c], i] = 1.0
    return arr


def pad_to_600bp(ohe_4x200: np.ndarray) -> np.ndarray:
    from experiments.train_malinois_k562 import MPRA_DOWNSTREAM, MPRA_UPSTREAM

    left = one_hot_encode(MPRA_UPSTREAM[-200:])
    right = one_hot_encode(MPRA_DOWNSTREAM[:200])
    return np.concatenate([left, ohe_4x200, right], axis=1)


def predict_with_legnet(seqs, checkpoint_dir, device="cuda"):
    """Predict using a LegNet ensemble checkpoint."""
    from models.legnet import LegNet

    ckpt_path = Path(checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(checkpoint_dir) / "last_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle ensemble format
    if "model_state_dicts" in ckpt:
        state_dicts = ckpt["model_state_dicts"]
    else:
        state_dicts = [ckpt.get("model_state_dict", ckpt)]

    preds_all = []
    for sd in state_dicts:
        model = LegNet()
        model.load_state_dict(sd)
        model.to(device).eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(seqs), 256):
                batch = seqs[i : i + 256]
                encoded = np.stack([one_hot_encode(s) for s in batch])
                x = torch.from_numpy(encoded).float().to(device)
                out = model(x)
                x_rc = x.flip(-1)[:, [3, 2, 1, 0], :]
                out_rc = model(x_rc)
                avg = ((out + out_rc) / 2).cpu().numpy().reshape(-1)
                preds.append(avg)
        preds_all.append(np.concatenate(preds))
        del model

    return np.mean(preds_all, axis=0)


def predict_with_malinois(seqs, device="cuda"):
    """Predict using pretrained Malinois (K562 output = index 0)."""
    from scripts.eval_pretrained_malinois import load_pretrained_malinois

    model = load_pretrained_malinois("data/pretrained/malinois_trained/torch_checkpoint.pt", device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            batch = seqs[i : i + 256]
            encoded = np.stack([pad_to_600bp(one_hot_encode(s)) for s in batch])
            x = torch.from_numpy(encoded).float().to(device)
            out = model(x)[:, 0]
            x_rc = x.flip(-1)[:, [3, 2, 1, 0], :]
            out_rc = model(x_rc)[:, 0]
            avg = ((out + out_rc) / 2).cpu().numpy().reshape(-1)
            preds.append(avg)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path("outputs/control_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load control sequences
    controls_path = Path("data/agarwal_2025/k562_all_controls_200bp.tsv")
    if not controls_path.exists():
        logger.error("Controls file not found: %s", controls_path)
        sys.exit(1)

    shuffled, ernst_neg, ernst_pos = [], [], []
    with open(controls_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["category"] == "shuffled_negative":
                shuffled.append(row["sequence"])
            elif row["category"] == "ernst_negative":
                ernst_neg.append(row["sequence"])
            elif row["category"] == "ernst_positive":
                ernst_pos.append(row["sequence"])

    logger.info(
        "Controls: %d shuffled, %d Ernst neg, %d Ernst pos",
        len(shuffled),
        len(ernst_neg),
        len(ernst_pos),
    )

    # Random sequences (same seed as bias comparison)
    rng = np.random.default_rng(42)
    random_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(10000)]

    all_seqs = {
        "shuffled_neg": shuffled,
        "ernst_neg": ernst_neg,
        "ernst_pos": ernst_pos,
        "random": random_seqs[:1000],  # Use 1K for speed
    }

    results = {}

    # 1. AG S2 Oracle
    logger.info("Loading AG S2 oracle...")
    try:
        from experiments.exp1_1_scaling import _load_oracle

        oracle = _load_oracle("k562", oracle_type="ag_s2")
        for cat, seqs in all_seqs.items():
            preds = oracle.predict(seqs)
            results.setdefault(cat, {})["AG S2"] = preds
            logger.info("  AG S2 %s: mean=%.3f", cat, np.mean(preds))
        del oracle
    except Exception as e:
        logger.warning("AG S2 failed: %s", e)

    # 2. Malinois (pretrained)
    logger.info("Loading Malinois...")
    try:
        for cat, seqs in all_seqs.items():
            preds = predict_with_malinois(seqs, device)
            results.setdefault(cat, {})["Malinois"] = preds
            logger.info("  Malinois %s: mean=%.3f", cat, np.mean(preds))
    except Exception as e:
        logger.warning("Malinois failed: %s", e)

    # 3. LegNet oracle
    logger.info("Loading LegNet oracle...")
    try:
        oracle = _load_oracle("k562", oracle_type="legnet")
        for cat, seqs in all_seqs.items():
            preds = oracle.predict(seqs)
            results.setdefault(cat, {})["LegNet Oracle"] = preds
            logger.info("  LegNet Oracle %s: mean=%.3f", cat, np.mean(preds))
        del oracle
    except Exception as e:
        logger.warning("LegNet oracle failed: %s", e)

    # 4. DREAM-RNN oracle
    logger.info("Loading DREAM-RNN oracle...")
    try:
        oracle = _load_oracle("k562", oracle_type="dream_rnn")
        for cat, seqs in all_seqs.items():
            preds = oracle.predict(seqs)
            results.setdefault(cat, {})["DREAM-RNN"] = preds
            logger.info("  DREAM-RNN %s: mean=%.3f", cat, np.mean(preds))
        del oracle
    except Exception as e:
        logger.warning("DREAM-RNN oracle failed: %s", e)

    # 5. LegNet (real labels, chr-split)
    logger.info("Loading LegNet (real labels)...")
    try:
        ckpt_dir = Path("outputs/legnet_k562_chr_split/seed_0")
        if ckpt_dir.exists():
            for cat, seqs in all_seqs.items():
                preds = predict_with_legnet(seqs, ckpt_dir, device)
                results.setdefault(cat, {})["LegNet (real)"] = preds
                logger.info("  LegNet real %s: mean=%.3f", cat, np.mean(preds))
    except Exception as e:
        logger.warning("LegNet real failed: %s", e)

    # Save results
    save_data = {}
    for cat, model_preds in results.items():
        for model, preds in model_preds.items():
            key = f"{cat}__{model.replace(' ', '_')}"
            save_data[key] = preds

    np.savez_compressed(out_dir / "control_predictions.npz", **save_data)

    # Summary
    print("\n" + "=" * 70)
    print("CONTROL SEQUENCE PREDICTION SUMMARY")
    print("=" * 70)
    categories = ["shuffled_neg", "ernst_neg", "ernst_pos", "random"]
    cat_labels = {
        "shuffled_neg": "Dinuc-shuffled (n=250)",
        "ernst_neg": "Ernst neg ctrl (n=200)",
        "ernst_pos": "Ernst pos ctrl (n=50)",
        "random": "Random DNA (n=1000)",
    }
    models = sorted(
        set(m for cat_d in results.values() for m in cat_d),
        key=lambda x: ["AG S2", "Malinois", "DREAM-RNN", "LegNet Oracle", "LegNet (real)"].index(x)
        if x in ["AG S2", "Malinois", "DREAM-RNN", "LegNet Oracle", "LegNet (real)"]
        else 99,
    )

    header = f"{'Category':<25s}" + "".join(f"{m:>15s}" for m in models)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat_labels.get(cat, cat):<25s}"
        for m in models:
            if cat in results and m in results[cat]:
                row += f"{np.mean(results[cat][m]):>15.3f}"
            else:
                row += f"{'N/A':>15s}"
        print(row)

    print(f"\nExpected: shuffled & random ≈ 0, Ernst pos > 0, Ernst neg ≈ 0")
    print(f"Saved to {out_dir / 'control_predictions.npz'}")

    # Generate figure
    _plot_results(results, categories, cat_labels, models, out_dir)


def _plot_results(results, categories, cat_labels, models, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "AG S2": "#1B5E20",
        "Malinois": "#1565C0",
        "DREAM-RNN": "#7B2D8E",
        "LegNet Oracle": "#D4A017",
        "LegNet (real)": "#E8602C",
    }

    fig, axes = plt.subplots(1, len(categories), figsize=(16, 5), sharey=True)

    for ax, cat in zip(axes, categories):
        if cat not in results:
            continue
        data = []
        clrs = []
        labels = []
        for m in models:
            if m in results[cat]:
                data.append(results[cat][m])
                clrs.append(colors.get(m, "#666"))
                labels.append(m)

        if data:
            bp = ax.boxplot(
                data,
                labels=labels,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="white", linewidth=2),
            )
            for patch, c in zip(bp["boxes"], clrs):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)

        ax.axhline(0, color="k", ls="--", lw=2, alpha=0.5)
        ax.set_title(cat_labels.get(cat, cat), fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Predicted K562 log2FC", fontsize=12)

    fig.suptitle(
        "Model Predictions on Control Sequences vs Random DNA\n"
        "(Agarwal et al. 2025 K562 lentiMPRA controls)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "control_vs_random_predictions.png", dpi=200, bbox_inches="tight")
    fig.savefig(
        str(Path("results/peter_figures/control_vs_random_predictions.png")),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved figure: {out_dir / 'control_vs_random_predictions.png'}")


if __name__ == "__main__":
    main()
