#!/usr/bin/env python
"""Evaluate NEW oracle ensemble on ALL test sets (full 35K SNV, 40K in-dist, 22K OOD).

Generates comprehensive landscape analysis figures comparing oracle vs real labels.
Uses the full 10-fold ensemble for predictions.

Usage:
    uv run --no-sync python scripts/eval_new_oracle_full.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    OUT = REPO / "results" / "peter_figures"
    OUT.mkdir(parents=True, exist_ok=True)

    # Load oracle
    from experiments.exp1_1_scaling import _load_oracle

    logger.info("Loading AG S2 oracle ensemble...")
    oracle = _load_oracle("k562", oracle_type="ag_s2")

    # ── In-dist (full hashfrag test, 40K) ──
    from data.k562 import K562Dataset

    ds = K562Dataset(data_path=str(REPO / "data" / "k562"), split="test")
    id_seqs = list(ds.sequences)
    id_true = ds.labels.astype(np.float32)
    logger.info("In-dist: %d sequences" % len(id_seqs))
    id_preds = oracle.predict(id_seqs)
    r_id = float(pearsonr(id_preds, id_true)[0])
    logger.info("  r=%.4f" % r_id)

    # ── SNV (full hashfrag, 35K pairs) ──
    snv_df = pd.read_csv(
        REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv", sep="\t"
    )
    ref_seqs = snv_df["sequence_ref"].tolist()
    alt_seqs = snv_df["sequence_alt"].tolist()
    true_delta = snv_df.get("delta_log2FC", snv_df.get("delta_label", pd.Series(dtype=float)))
    true_delta = true_delta.values.astype(np.float32) if len(true_delta) > 0 else None
    true_alt = snv_df.get("K562_log2FC_alt", pd.Series(dtype=float))
    true_alt = true_alt.values.astype(np.float32) if len(true_alt) > 0 else None

    logger.info("SNV: %d pairs" % len(ref_seqs))
    ref_preds = oracle.predict(ref_seqs)
    alt_preds = oracle.predict(alt_seqs)
    delta_preds = alt_preds - ref_preds
    if true_delta is not None:
        r_delta = float(pearsonr(delta_preds, true_delta)[0])
        dir_acc = float((np.sign(delta_preds) == np.sign(true_delta)).mean())
        logger.info("  delta r=%.4f, direction=%.1f%%" % (r_delta, 100 * dir_acc))

    # ── OOD (22K designed) ──
    ood_df = pd.read_csv(
        REPO / "data" / "k562" / "test_sets" / "test_ood_designed_k562.tsv", sep="\t"
    )
    ood_seqs = ood_df["sequence"].tolist()
    ood_true = ood_df["K562_log2FC"].values.astype(np.float32)
    logger.info("OOD: %d sequences" % len(ood_seqs))
    ood_preds = oracle.predict(ood_seqs)
    r_ood = float(pearsonr(ood_preds, ood_true)[0])
    logger.info("  r=%.4f" % r_ood)

    # ── Random sequences (check oracle bias) ──
    rng = np.random.default_rng(42)
    rand_seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(10000)]
    rand_preds = oracle.predict(rand_seqs)
    logger.info("Random 10K: mean=%.3f std=%.3f" % (np.mean(rand_preds), np.std(rand_preds)))

    # ── Save predictions ──
    ens_dir = REPO / "outputs" / "oracle_full_856k" / "ensemble_eval"
    ens_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ens_dir / "all_test_predictions.npz",
        id_preds=id_preds,
        id_true=id_true,
        ref_preds=ref_preds,
        alt_preds=alt_preds,
        delta_preds=delta_preds,
        true_alt=true_alt if true_alt is not None else np.array([]),
        true_delta=true_delta if true_delta is not None else np.array([]),
        ood_preds=ood_preds,
        ood_true=ood_true,
        rand_preds=rand_preds,
    )

    # ══════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A: In-dist scatter
    ax = axes[0, 0]
    ax.scatter(id_true, id_preds, s=0.5, alpha=0.1, color="#1B5E20")
    ax.plot([-3, 10], [-3, 10], "k--", lw=0.5)
    ax.text(
        0.05,
        0.95,
        "In-dist (n=%d)\nr = %.4f\nbias = %+.3f"
        % (len(id_true), r_id, float(np.mean(id_preds - id_true))),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("True K562 log2FC")
    ax.set_ylabel("Oracle ensemble prediction")
    ax.set_title("A. In-Distribution", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # B: SNV delta scatter (full 35K)
    ax = axes[0, 1]
    if true_delta is not None:
        ax.scatter(true_delta, delta_preds, s=0.5, alpha=0.15, color="#1B5E20")
        lim = 3
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
        slope = np.polyfit(true_delta, delta_preds, 1)[0]
        ax.plot([-lim, lim], [-lim * slope, lim * slope], "r-", lw=1, alpha=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.text(
            0.05,
            0.95,
            "SNV delta (n=%d)\nr = %.4f\ndirection = %.1f%%\nslope = %.2f\nstd ratio = %.3f"
            % (
                len(true_delta),
                r_delta,
                100 * dir_acc,
                slope,
                np.std(delta_preds) / np.std(true_delta),
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8),
        )
    ax.set_xlabel("True SNV delta")
    ax.set_ylabel("Oracle SNV delta")
    ax.set_title("B. SNV Effect (Full 35K pairs)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # C: OOD scatter
    ax = axes[0, 2]
    ax.scatter(ood_true, ood_preds, s=0.5, alpha=0.1, color="#d62728")
    ax.plot([-3, 10], [-3, 10], "k--", lw=0.5)
    ax.text(
        0.05,
        0.95,
        "OOD designed (n=%d)\nr = %.4f\nbias = %+.3f"
        % (len(ood_true), r_ood, float(np.mean(ood_preds - ood_true))),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("True K562 log2FC")
    ax.set_ylabel("Oracle ensemble prediction")
    ax.set_title("C. Out-of-Distribution", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # D: Oracle bias on random sequences
    ax = axes[1, 0]
    bins = np.linspace(-2, 8, 80)
    ax.hist(id_preds, bins=bins, alpha=0.5, color="#1B5E20", label="In-dist oracle", density=True)
    ax.hist(
        rand_preds,
        bins=bins,
        alpha=0.5,
        color="#888888",
        label="Random oracle\n(mean=%.2f)" % np.mean(rand_preds),
        density=True,
    )
    ax.hist(id_true, bins=bins, alpha=0.3, color="#333333", label="In-dist real", density=True)
    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("K562 log2FC")
    ax.set_ylabel("Density")
    ax.set_title("D. Oracle Bias: Random DNA", fontsize=11)
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # E: SNV direction accuracy by effect size
    ax = axes[1, 1]
    if true_delta is not None:
        thresholds = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        centers, accs, ns = [], [], []
        for i in range(len(thresholds)):
            if i < len(thresholds) - 1:
                lo, hi = thresholds[i], thresholds[i + 1]
            else:
                lo, hi = thresholds[i], 10.0
            mask = (np.abs(true_delta) >= lo) & (np.abs(true_delta) < hi)
            if mask.sum() > 10:
                acc = float((np.sign(delta_preds[mask]) == np.sign(true_delta[mask])).mean())
                centers.append("%.1f-%.1f" % (lo, hi) if hi < 10 else ">%.1f" % lo)
                accs.append(100 * acc)
                ns.append(mask.sum())
        bars = ax.bar(range(len(centers)), accs, color="#1B5E20", alpha=0.7)
        ax.axhline(50, color="k", ls="--", lw=1, label="Chance")
        ax.set_xticks(range(len(centers)))
        ax.set_xticklabels(centers, fontsize=7, rotation=30)
        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                "n=%d" % n,
                ha="center",
                fontsize=5,
            )
        ax.set_xlabel("|True delta| range")
        ax.set_ylabel("Direction accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
    ax.set_title("E. SNV Direction by Effect Size", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # F: Compression ratio by effect size
    ax = axes[1, 2]
    if true_delta is not None:
        abs_true = np.abs(true_delta)
        abs_pred = np.abs(delta_preds)
        n_bins = 15
        bin_edges = np.percentile(abs_true[abs_true > 0.01], np.linspace(0, 100, n_bins + 1))
        ctrs, ratios = [], []
        for i in range(n_bins):
            mask = (abs_true >= bin_edges[i]) & (abs_true < bin_edges[i + 1])
            if mask.sum() > 10:
                ctrs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                ratios.append(np.mean(abs_pred[mask]) / np.mean(abs_true[mask]))
        ax.bar(
            ctrs,
            ratios,
            width=np.diff(bin_edges[: len(ctrs) + 1]) * 0.8,
            color="#1B5E20",
            alpha=0.7,
        )
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("|True delta| magnitude")
        ax.set_ylabel("Oracle / Real ratio")
        ax.text(
            0.95,
            0.95,
            "< 1 = oracle\nunderpredicts",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            style="italic",
        )
    ax.set_title("F. Compression by Effect Size", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "NEW AG S2 Oracle Ensemble (856K training) — Full Test Set Evaluation",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "new_oracle_full_landscape.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "new_oracle_full_landscape.pdf", dpi=200, bbox_inches="tight")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  In-dist (40K):  r=%.4f" % r_id)
    if true_delta is not None:
        print("  SNV delta (35K): r=%.4f, direction=%.1f%%" % (r_delta, 100 * dir_acc))
    print("  OOD (22K):      r=%.4f" % r_ood)
    print("  Random bias:    mean=%.3f (should be ~0)" % np.mean(rand_preds))
    print("\n  Saved: new_oracle_full_landscape.png")


if __name__ == "__main__":
    main()
