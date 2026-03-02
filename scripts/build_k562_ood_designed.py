#!/usr/bin/env python3
"""
Build the K562 OOD test set from OL46 designed sequences (Gosai et al. 2024).

Source: DATA-MPRA_Datasets.zip from Zenodo record 10698014
  - MPRA_Datasets/Validation/OL46.attributes      (design method + target cell)
  - MPRA_Datasets/Validation/OL46_K562_20220720.out  (K562 expression measurements)
  - MPRA_Datasets/Fastas/OL46_reference.fasta.gz     (200bp sequences)

Filters to K562-targeting sequences from the three CODA design algorithms:
  - BODA:fsp:k562  (FastSeqProp → 9,000 sequences)
  - BODA:sa:k562   (Simulated Annealing → 9,000 sequences)
  - BODA:al:k562   (AdaLead → 5,000 sequences)

Output: data/k562/test_sets/test_ood_designed_k562.tsv
  Columns: ID, sequence, method, target_cell, K562_log2FC, K562_lfcSE

Usage (run from repo root, after downloading/extracting OL46 data):
  python scripts/build_k562_ood_designed.py \\
      --attrs   /path/to/MPRA_Datasets/Validation/OL46.attributes \\
      --k562    /path/to/MPRA_Datasets/Validation/OL46_K562_20220720.out \\
      --fasta   /path/to/MPRA_Datasets/Fastas/OL46_reference.fasta.gz \\
      --out     data/k562/test_sets/test_ood_designed_k562.tsv
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

# K562-targeting designed-sequence project labels in OL46
K562_DESIGNED_PROJECTS = {"BODA:fsp:k562", "BODA:sa:k562", "BODA:al:k562"}

# Map project prefix → readable method name
METHOD_MAP = {
    "BODA:fsp": "FastSeqProp",
    "BODA:sa": "Simulated_Annealing",
    "BODA:al": "AdaLead",
}


def load_fasta(path: Path) -> dict[str, str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    fasta: dict[str, str] = {}
    current_id = None
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                current_id = line[1:]
                fasta[current_id] = []
            elif current_id is not None:
                fasta[current_id].append(line)
    return {k: "".join(v) for k, v in fasta.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attrs",
        type=Path,
        default=Path("/tmp/ol46_data/MPRA_Datasets/Validation/OL46.attributes"),
    )
    parser.add_argument(
        "--k562",
        type=Path,
        default=Path("/tmp/ol46_data/MPRA_Datasets/Validation/OL46_K562_20220720.out"),
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("/tmp/ol46_data/MPRA_Datasets/Fastas/OL46_reference.fasta.gz"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/k562/test_sets/test_ood_designed_k562.tsv"),
    )
    args = parser.parse_args()

    # 1. Load attributes, filter to K562-targeting designed sequences
    print("Loading OL46 attributes ...")
    attrs = pd.read_csv(args.attrs, sep="\t", low_memory=False)
    print(f"  Total entries: {len(attrs):,}")

    k562_mask = attrs["project"].isin(K562_DESIGNED_PROJECTS)
    attrs_k562 = attrs[k562_mask][["ID", "project"]].copy()
    attrs_k562["method"] = attrs_k562["project"].map(
        lambda p: next((v for k, v in METHOD_MAP.items() if p.startswith(k)), p)
    )
    attrs_k562["target_cell"] = "k562"
    print(f"  K562-designed (fsp/sa/al): {len(attrs_k562):,}")
    print(f"  {attrs_k562['method'].value_counts().to_dict()}")

    # 2. Load K562 expression
    print("Loading OL46 K562 expression ...")
    k562_exp = pd.read_csv(args.k562, sep="\t", low_memory=False)
    k562_exp = k562_exp.rename(columns={"log2FoldChange": "K562_log2FC", "lfcSE": "K562_lfcSE"})[
        ["ID", "K562_log2FC", "K562_lfcSE"]
    ]
    print(f"  Expression rows: {len(k562_exp):,}")

    # 3. Load FASTA sequences
    print("Loading OL46 reference FASTA ...")
    fasta = load_fasta(args.fasta)
    print(f"  FASTA entries: {len(fasta):,}")

    # 4. Join
    df = attrs_k562.merge(k562_exp, on="ID", how="inner")
    print(f"  After joining with expression: {len(df):,}")

    df["sequence"] = df["ID"].map(fasta)
    missing_seq = df["sequence"].isna().sum()
    if missing_seq > 0:
        print(f"  WARNING: {missing_seq} sequences missing from FASTA; dropping.")
    df = df.dropna(subset=["sequence", "K562_log2FC"])

    # 5. Validate sequence lengths
    df["seq_len"] = df["sequence"].str.len()
    wrong_len = (df["seq_len"] != 200).sum()
    if wrong_len > 0:
        print(f"  WARNING: {wrong_len} sequences are not 200bp; dropping.")
    df = df[df["seq_len"] == 200].drop(columns=["seq_len"])

    print(f"\nFinal OOD set: {len(df):,} sequences")
    print(df["method"].value_counts().to_dict())

    # 6. Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["ID", "sequence", "method", "target_cell", "K562_log2FC", "K562_lfcSE"]
    df[out_cols].to_csv(args.out, sep="\t", index=False)
    print(f"Wrote: {args.out}")

    # 7. Update manifest
    manifest_path = args.out.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}
    manifest["ood_designed_file"] = str(args.out)
    manifest["ood_designed_criteria"] = {
        "source": "Gosai et al. 2024, Nature — OL46 Validation library",
        "zenodo": "10698014 / DATA-MPRA_Datasets.zip",
        "projects": sorted(K562_DESIGNED_PROJECTS),
        "methods": sorted(METHOD_MAP.values()),
        "n": len(df),
        "counts": df["method"].value_counts().to_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
