"""Causal CpG intervention test on the AG_S2 oracle.

For each of ~1000 real Gosai sequences spanning the activity range:
  - Original sequence
  - CpG-depleted variant: replace each CG with TG (mutates ~3 CpGs per seq on average)
  - CpG-enriched variant: insert CGs into AT runs at non-motif positions

Predict each with AG_S2 oracle. If the predictions track the CpG manipulation
(predicted activity drops when CpG removed, rises when added), the model learned
a CpG → activity relationship that's *responsive* (and therefore acts causally
on novel inputs even if it's confounded in training).

If predictions barely change, the model's CpG-correlation is correlational
only and we can't blame CpG content alone for random-DNA overprediction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
OUT = REPO / "outputs/cpg_causal_intervention"


def cpg_density(seq: str) -> float:
    seq = seq.upper()
    if len(seq) <= 1:
        return 0.0
    return sum(1 for i in range(len(seq) - 1) if seq[i] == "C" and seq[i + 1] == "G") / (
        len(seq) - 1
    )


def cpg_deplete(seq: str) -> str:
    """Replace each CG with TG."""
    out = list(seq.upper())
    for i in range(len(out) - 1):
        if out[i] == "C" and out[i + 1] == "G":
            out[i] = "T"
    return "".join(out)


def cpg_enrich(seq: str, n_add: int, rng) -> str:
    """Insert n_add CG dinucleotides into AT runs at random positions."""
    out = list(seq.upper())
    # Find AT positions (i, i+1) where both are A or T
    at_pairs = [i for i in range(len(out) - 1) if out[i] in "AT" and out[i + 1] in "AT"]
    if not at_pairs:
        return seq
    n_add = min(n_add, len(at_pairs))
    chosen = rng.choice(at_pairs, size=n_add, replace=False)
    for pos in chosen:
        out[pos] = "C"
        out[pos + 1] = "G"
    return "".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Gosai data...")
    df = pd.read_csv(REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt", sep="\t", low_memory=False)
    df = df.dropna(subset=["K562_log2FC", "sequence"]).copy()

    # Stratified sample by activity decile (n=100 per decile = 1000 total)
    rng = np.random.default_rng(42)
    df["activity_q"] = pd.qcut(df["K562_log2FC"], 10, labels=False)
    sampled = []
    for q in range(10):
        sub = df[df["activity_q"] == q]
        if len(sub) > 100:
            sub = sub.sample(n=100, random_state=42)
        sampled.append(sub)
    sample = pd.concat(sampled, ignore_index=True)
    logger.info(f"  stratified sample n={len(sample):,}")

    # Build the 3 variants
    logger.info("Constructing CpG-depleted and CpG-enriched variants...")
    sample["seq_original"] = sample["sequence"]
    sample["seq_depleted"] = sample["seq_original"].apply(cpg_deplete)
    sample["seq_enriched"] = sample.apply(
        lambda r: cpg_enrich(r["seq_original"], n_add=8, rng=rng), axis=1
    )
    sample["cpg_original"] = sample["seq_original"].apply(cpg_density)
    sample["cpg_depleted"] = sample["seq_depleted"].apply(cpg_density)
    sample["cpg_enriched"] = sample["seq_enriched"].apply(cpg_density)
    logger.info(
        f"  CpG density: original μ={sample['cpg_original'].mean():.4f}, "
        f"depleted μ={sample['cpg_depleted'].mean():.4f}, "
        f"enriched μ={sample['cpg_enriched'].mean():.4f}"
    )

    # Score all 3 variants with AG_S2 oracle
    from experiments.exp1_1_scaling import _load_oracle

    logger.info("Loading AG_S2 oracle...")
    oracle = _load_oracle("k562", oracle_type="ag_s2")

    logger.info("Scoring original sequences...")
    pred_original = oracle.predict(sample["seq_original"].tolist()).astype(np.float32)
    logger.info("Scoring CpG-depleted sequences...")
    pred_depleted = oracle.predict(sample["seq_depleted"].tolist()).astype(np.float32)
    logger.info("Scoring CpG-enriched sequences...")
    pred_enriched = oracle.predict(sample["seq_enriched"].tolist()).astype(np.float32)

    sample["pred_original"] = pred_original
    sample["pred_depleted"] = pred_depleted
    sample["pred_enriched"] = pred_enriched
    sample["d_depleted"] = pred_depleted - pred_original
    sample["d_enriched"] = pred_enriched - pred_original
    sample["d_cpg_depleted"] = sample["cpg_depleted"] - sample["cpg_original"]
    sample["d_cpg_enriched"] = sample["cpg_enriched"] - sample["cpg_original"]

    # Save
    sample[
        [
            "pair_id",
            "K562_log2FC",
            "activity_q",
            "cpg_original",
            "cpg_depleted",
            "cpg_enriched",
            "pred_original",
            "pred_depleted",
            "pred_enriched",
            "d_depleted",
            "d_enriched",
            "d_cpg_depleted",
            "d_cpg_enriched",
        ]
    ].to_csv(
        OUT / "cpg_intervention_results.csv", index=False
    ) if "pair_id" in sample.columns else sample[
        [
            c
            for c in sample.columns
            if c not in ("seq_original", "seq_depleted", "seq_enriched", "sequence")
        ]
    ].to_csv(OUT / "cpg_intervention_results.csv", index=False)

    # Summary
    logger.info("\n=== Causal intervention results ===")
    logger.info(
        f"\nMean Δpred when CpGs DEPLETED: {sample['d_depleted'].mean():+.4f} (std {sample['d_depleted'].std():.3f})"
    )
    logger.info(
        f"Mean Δpred when CpGs ENRICHED: {sample['d_enriched'].mean():+.4f} (std {sample['d_enriched'].std():.3f})"
    )
    logger.info(f"Mean ΔCpG (deplete):    {sample['d_cpg_depleted'].mean():+.4f}")
    logger.info(f"Mean ΔCpG (enrich):     {sample['d_cpg_enriched'].mean():+.4f}")

    # Implied causal slope
    slope_deplete = (
        sample["d_depleted"].mean() / sample["d_cpg_depleted"].mean()
        if abs(sample["d_cpg_depleted"].mean()) > 1e-6
        else float("nan")
    )
    slope_enrich = (
        sample["d_enriched"].mean() / sample["d_cpg_enriched"].mean()
        if abs(sample["d_cpg_enriched"].mean()) > 1e-6
        else float("nan")
    )
    logger.info(
        f"\nImplied CpG causal slope (deplete arm): {slope_deplete:+.2f} log2FC per unit CpG"
    )
    logger.info(f"Implied CpG causal slope (enrich arm):  {slope_enrich:+.2f}")
    logger.info(
        f"  (compare to: real-data slope in natural Gosai ≈ +23, oracle titration slope ≈ +13)"
    )

    # Per activity decile
    logger.info(f"\n=== Δpred per activity decile (depletion arm) ===")
    logger.info(
        f"{'q':>3}  {'K562 mean':>10}  {'pred_orig mean':>14}  {'Δpred dep':>10}  {'Δpred enr':>10}"
    )
    for q in range(10):
        sub = sample[sample["activity_q"] == q]
        logger.info(
            f"{q:>3}  {sub['K562_log2FC'].mean():>+10.3f}  {sub['pred_original'].mean():>+14.3f}  "
            f"{sub['d_depleted'].mean():>+10.3f}  {sub['d_enriched'].mean():>+10.3f}"
        )

    import json

    summary = {
        "n": len(sample),
        "mean_d_depleted": float(sample["d_depleted"].mean()),
        "mean_d_enriched": float(sample["d_enriched"].mean()),
        "implied_causal_slope_deplete": float(slope_deplete)
        if not np.isnan(slope_deplete)
        else None,
        "implied_causal_slope_enrich": float(slope_enrich) if not np.isnan(slope_enrich) else None,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
