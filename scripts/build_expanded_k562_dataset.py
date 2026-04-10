#!/usr/bin/env python
"""Build expanded K562 dataset: Gosai Table S2 + Agarwal intergenic + controls.

Merges the existing Gosai training data with Agarwal et al. 2025 intergenic
sequences and controls to recalibrate model baselines.

Steps:
1. Load Gosai Table S2 (existing training data)
2. Load Agarwal ENCODE element quantifications (ENCFF252GNM)
3. Load Agarwal S3 library design for categories
4. Extract intergenic sequences and controls with activity
5. Match sequences from S3 to activity values
6. Split new sequences into train/test (90/10) stratified by category
7. Save expanded dataset

Usage:
    uv run --no-sync python scripts/build_expanded_k562_dataset.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    out_dir = REPO / "data" / "k562_expanded"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # 1. Load existing Gosai Table S2
    # ═══════════════════════════════════════════════════════
    logger.info("Loading Gosai Table S2...")
    import pandas as pd

    gosai_path = REPO / "data" / "k562" / "DATA-Table_S2__MPRA_dataset.txt"
    gosai = pd.read_csv(gosai_path, sep="\t")
    logger.info("  Gosai: %d rows", len(gosai))
    logger.info(
        "  K562_log2FC: mean=%.4f, std=%.4f",
        gosai["K562_log2FC"].mean(),
        gosai["K562_log2FC"].std(),
    )

    # Build set of existing Gosai sequences for dedup
    gosai_seqs = set(gosai["sequence"].str.upper().tolist())
    logger.info("  Unique sequences: %d", len(gosai_seqs))

    # ═══════════════════════════════════════════════════════
    # 2. Load Agarwal ENCODE quantifications
    # ═══════════════════════════════════════════════════════
    logger.info("Loading Agarwal ENCODE quantifications...")
    encode_path = REPO / "data" / "agarwal_2025" / "ENCFF252GNM.tsv"
    data_by_name = defaultdict(list)
    with open(encode_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["name"] != "no_BC":
                data_by_name[row["name"]].append(float(row["log2"]))

    encode_mean = {n: np.mean(v) for n, v in data_by_name.items()}
    logger.info("  ENCODE elements with activity: %d", len(encode_mean))

    # ═══════════════════════════════════════════════════════
    # 3. Load S3 library design (sequences + categories)
    # ═══════════════════════════════════════════════════════
    logger.info("Loading Agarwal S3 library design...")
    s3_path = REPO / "data" / "agarwal_2025" / "Table_S3_large_scale_lib_design.xlsx"
    wb = openpyxl.load_workbook(s3_path, read_only=True)
    ws = wb["K562 large-scale"]

    s3_elements = {}  # name -> {category, sequence_200bp}
    for row in ws.iter_rows(min_row=3, values_only=True):
        name, cat = row[0], row[1]
        seq_230 = row[6] if len(row) > 6 else None
        if name and cat and seq_230 and len(seq_230) >= 230:
            seq_200 = seq_230[15:-15]  # strip 15nt adaptors
            s3_elements[str(name)] = {"category": str(cat), "sequence": seq_200}

    logger.info("  S3 elements: %d", len(s3_elements))

    # ═══════════════════════════════════════════════════════
    # 4. Extract new elements (intergenic + controls)
    # ═══════════════════════════════════════════════════════
    target_cats = {
        "intergenic (7 loci)",
        "negative control, shuffled",
        "negative control (Ernst et al 2016)",
        "positive control (Ernst et al 2016)",
    }

    new_elements = []
    for name, info in s3_elements.items():
        if info["category"] not in target_cats:
            continue
        seq = info["sequence"].upper()
        if len(seq) != 200:
            continue

        # Get activity (average forward + reverse if both exist)
        activities = []
        for suffix in ["", "_Reversed:"]:
            lookup = name + suffix
            if lookup in encode_mean:
                activities.append(encode_mean[lookup])

        if not activities:
            continue

        # Use forward-orientation activity if available, else average
        fwd_val = encode_mean.get(name, None)
        activity = fwd_val if fwd_val is not None else np.mean(activities)

        # Check if already in Gosai
        if seq in gosai_seqs:
            continue

        new_elements.append(
            {
                "name": name,
                "category": info["category"],
                "sequence": seq,
                "K562_log2FC_encode_raw": activity,
            }
        )

    logger.info("New elements to add (not in Gosai):")
    cat_counts = defaultdict(int)
    for el in new_elements:
        cat_counts[el["category"]] += 1
    for cat, n in sorted(cat_counts.items()):
        logger.info("  %s: %d", cat, n)

    # ═══════════════════════════════════════════════════════
    # 5. Normalize new elements to match Gosai scale
    # ═══════════════════════════════════════════════════════
    # The Gosai K562_log2FC values use DESeq2 normalization.
    # The ENCODE raw log2(RNA/DNA) is on a different scale.
    # We need to map ENCODE -> Gosai scale.
    #
    # Strategy: find overlapping elements (same name pattern)
    # and compute a linear mapping.
    #
    # Since we can't easily match by name (different ID schemes),
    # we'll use the overall distribution statistics:
    # - Gosai enhancers: mean≈1.87, std≈1.85 (from Table S2)
    # - ENCODE enhancers: mean≈-0.12, std≈0.49
    # This suggests Gosai = a * ENCODE + b
    #
    # Better approach: use the Agarwal S6 as intermediate
    # (z-scored), then map to Gosai.
    #
    # For now, we'll use a simpler approach: the ENCODE raw values
    # capture the relative differences correctly. We add new elements
    # using ENCODE raw scale and note this in the metadata.
    # The model training can handle mixed scales by normalizing.
    #
    # Actually, the cleanest approach: re-normalize everything to
    # have the same mean/std as Gosai K562_log2FC.

    logger.info("Computing normalization mapping...")
    gosai_mean = gosai["K562_log2FC"].mean()
    gosai_std = gosai["K562_log2FC"].std()

    # Get ENCODE values for elements likely in Gosai (enhancers + promoters)
    encode_genomic = []
    for name, info in s3_elements.items():
        if info["category"] in ("potential enhancer", "promoter"):
            if name in encode_mean:
                encode_genomic.append(encode_mean[name])
    encode_genomic = np.array(encode_genomic)
    enc_mean = np.mean(encode_genomic)
    enc_std = np.std(encode_genomic)

    logger.info("  Gosai K562_log2FC: mean=%.4f, std=%.4f", gosai_mean, gosai_std)
    logger.info("  ENCODE genomic: mean=%.4f, std=%.4f", enc_mean, enc_std)

    # Linear transform: gosai_scale = (encode_raw - enc_mean) / enc_std * gosai_std + gosai_mean
    def encode_to_gosai(val):
        return (val - enc_mean) / enc_std * gosai_std + gosai_mean

    # Verify transform makes sense
    logger.info(
        "  Transform check: ENCODE 0.0 -> Gosai %.2f",
        encode_to_gosai(0.0),
    )
    logger.info(
        "  Transform check: ENCODE -0.53 (shuffled) -> Gosai %.2f",
        encode_to_gosai(-0.53),
    )
    logger.info(
        "  Transform check: ENCODE +0.50 (pos ctrl) -> Gosai %.2f",
        encode_to_gosai(0.50),
    )

    # Apply transform to new elements
    for el in new_elements:
        el["K562_log2FC"] = encode_to_gosai(el["K562_log2FC_encode_raw"])

    # ═══════════════════════════════════════════════════════
    # 6. Split new elements: 90% train, 10% test
    # ═══════════════════════════════════════════════════════
    rng = np.random.default_rng(42)
    train_new = []
    test_new = []

    for cat in target_cats:
        cat_els = [el for el in new_elements if el["category"] == cat]
        rng.shuffle(cat_els)
        n_test = max(1, int(0.1 * len(cat_els)))
        test_new.extend(cat_els[:n_test])
        train_new.extend(cat_els[n_test:])

    logger.info("Split: %d train, %d test", len(train_new), len(test_new))

    # ═══════════════════════════════════════════════════════
    # 7. Save expanded dataset
    # ═══════════════════════════════════════════════════════

    # Save new train elements
    train_path = out_dir / "agarwal_new_train.tsv"
    with open(train_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "category",
                "sequence",
                "K562_log2FC",
                "K562_log2FC_encode_raw",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for el in train_new:
            writer.writerow(el)
    logger.info("Saved train: %s (%d elements)", train_path, len(train_new))

    # Save test elements
    test_path = out_dir / "agarwal_new_test.tsv"
    with open(test_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "category",
                "sequence",
                "K562_log2FC",
                "K562_log2FC_encode_raw",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for el in test_new:
            writer.writerow(el)
    logger.info("Saved test: %s (%d elements)", test_path, len(test_new))

    # Save metadata
    meta = {
        "gosai_n": len(gosai),
        "gosai_mean": float(gosai_mean),
        "gosai_std": float(gosai_std),
        "encode_genomic_mean": float(enc_mean),
        "encode_genomic_std": float(enc_std),
        "new_train_n": len(train_new),
        "new_test_n": len(test_new),
        "transform": f"gosai_scale = (encode_raw - {enc_mean:.4f}) / {enc_std:.4f} * {gosai_std:.4f} + {gosai_mean:.4f}",
        "categories": {cat: cat_counts[cat] for cat in target_cats},
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata: %s", meta_path)

    # Summary
    new_vals = np.array([el["K562_log2FC"] for el in new_elements])
    print("\n" + "=" * 60)
    print("EXPANDED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Gosai original: {len(gosai):,} sequences")
    print(f"  New elements:   {len(new_elements):,} sequences")
    print(f"    Train: {len(train_new):,}")
    print(f"    Test:  {len(test_new):,}")
    print(f"  Total training: {len(gosai) + len(train_new):,}")
    print(f"\n  New element K562_log2FC (Gosai scale):")
    print(f"    mean={np.mean(new_vals):.4f}, std={np.std(new_vals):.4f}")
    print(f"    range=[{np.min(new_vals):.4f}, {np.max(new_vals):.4f}]")

    for cat in sorted(target_cats):
        cat_vals = [el["K562_log2FC"] for el in new_elements if el["category"] == cat]
        if cat_vals:
            print(f"\n  {cat}:")
            print(
                f"    n={len(cat_vals)}, mean={np.mean(cat_vals):.4f}, "
                f"raw_mean={np.mean([el['K562_log2FC_encode_raw'] for el in new_elements if el['category'] == cat]):.4f}"
            )


if __name__ == "__main__":
    main()
