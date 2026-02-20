#!/usr/bin/env python3
"""Create standardized K562 evaluation test sets from available benchmark data.

Outputs under ``data/k562/test_sets``:
- ``test_in_distribution_hashfrag.tsv``: HashFrag test split (expression prediction).
- ``test_snv_pairs_hashfrag.tsv``: ref/alt SNV pairs from the HashFrag test split.
- ``test_ood_cre.tsv``: out-of-domain proxy set from the CRE library.
- ``manifest.json``: criteria and counts used to build each file.

Notes:
- The public Gosai K562 file in this repo does not include ``target_cell`` / ``origin``
  columns, so the exact AdaLead/FastSeqProp/Simulated_Annealing filter is unavailable.
- In this case, OOD is built from ``data_project == 'CRE'`` as the best available proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_ids(df: pd.DataFrame) -> pd.DataFrame:
    id_parts = df["IDs"].astype(str).str.split(":", expand=True)
    df = df.copy()
    df["chr_parsed"] = id_parts[0]
    df["pos_parsed"] = id_parts[1]
    df["ref_parsed"] = id_parts[2]
    df["alt_parsed"] = id_parts[3]
    df["allele_parsed"] = id_parts[4]
    return df


def _filtered_reference_like(df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_ids(df)
    is_reference = df["allele_parsed"] == "R"
    is_non_variant = (df["ref_parsed"] == "NA") & (df["alt_parsed"] == "NA")
    keep = df[is_reference | is_non_variant].copy()

    keep["seq_len"] = keep["sequence"].astype(str).str.len()
    keep = keep[keep["seq_len"] >= 198].copy()
    keep = keep.drop(columns=["seq_len"])
    keep = keep.reset_index(drop=True)
    return keep


def _build_in_distribution_hashfrag(
    filtered_df: pd.DataFrame, hashfrag_dir: Path, out_dir: Path
) -> Path:
    test_idx_path = hashfrag_dir / "test_indices.npy"
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Missing HashFrag test indices: {test_idx_path}")
    test_idx = np.load(test_idx_path)

    in_dist = filtered_df.iloc[test_idx].copy()
    out_path = out_dir / "test_in_distribution_hashfrag.tsv"
    in_dist.to_csv(out_path, sep="\t", index=False)
    return out_path


def _build_snv_pairs_from_hashfrag_test(in_dist_df: pd.DataFrame, out_dir: Path) -> Path:
    has_snv = (
        (in_dist_df["ref_parsed"] != "NA")
        & (in_dist_df["alt_parsed"] != "NA")
        & (in_dist_df["allele_parsed"].isin(["A", "R"]))
    )
    snv_df = in_dist_df[has_snv].copy()

    snv_ref = snv_df[snv_df["allele_parsed"] == "R"].copy()
    snv_alt = snv_df[snv_df["allele_parsed"] == "A"].copy()

    snv_ref["pair_key"] = (
        snv_ref["chr_parsed"]
        + ":"
        + snv_ref["pos_parsed"]
        + ":"
        + snv_ref["ref_parsed"]
        + ":"
        + snv_ref["alt_parsed"]
    )
    snv_alt["pair_key"] = (
        snv_alt["chr_parsed"]
        + ":"
        + snv_alt["pos_parsed"]
        + ":"
        + snv_alt["ref_parsed"]
        + ":"
        + snv_alt["alt_parsed"]
    )

    snv_pairs = pd.merge(
        snv_ref[["pair_key", "IDs", "sequence", "K562_log2FC", "K562_lfcSE"]],
        snv_alt[["pair_key", "IDs", "sequence", "K562_log2FC", "K562_lfcSE"]],
        on="pair_key",
        suffixes=("_ref", "_alt"),
    )
    snv_pairs["delta_log2FC"] = snv_pairs["K562_log2FC_alt"] - snv_pairs["K562_log2FC_ref"]

    out_path = out_dir / "test_snv_pairs_hashfrag.tsv"
    snv_pairs.to_csv(out_path, sep="\t", index=False)
    return out_path


def _build_ood_set(raw_df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    if "target_cell" in raw_df.columns and "origin" in raw_df.columns:
        ood_mask = raw_df["target_cell"].astype(str).str.lower().eq("k562") & raw_df[
            "origin"
        ].astype(str).isin(["AdaLead", "FastSeqProp", "Simulated_Annealing"])
        criteria = {
            "mode": "target_cell_origin",
            "target_cell": "k562",
            "origin": ["AdaLead", "FastSeqProp", "Simulated_Annealing"],
        }
    else:
        # Best available proxy from public columns.
        ood_mask = raw_df["data_project"].astype(str).eq("CRE")
        criteria = {
            "mode": "data_project_proxy",
            "data_project": "CRE",
            "note": "target_cell/origin columns unavailable in this dataset file",
        }

    ood_df = raw_df[ood_mask].copy()
    out_path = out_dir / "test_ood_cre.tsv"
    ood_df.to_csv(out_path, sep="\t", index=False)
    return out_path, criteria


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/k562"))
    args = parser.parse_args()

    data_root = args.data_root
    raw_file = data_root / "DATA-Table_S2__MPRA_dataset.txt"
    hashfrag_dir = data_root / "hashfrag_splits"
    if not raw_file.exists():
        raise SystemExit(f"Missing raw file: {raw_file}")

    out_dir = data_root / "test_sets"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_file, sep="\t", dtype={"OL": str})
    filtered_df = _filtered_reference_like(raw_df)

    in_dist_path = _build_in_distribution_hashfrag(
        filtered_df=filtered_df,
        hashfrag_dir=hashfrag_dir,
        out_dir=out_dir,
    )

    in_dist_df = pd.read_csv(in_dist_path, sep="\t")
    snv_path = _build_snv_pairs_from_hashfrag_test(in_dist_df=in_dist_df, out_dir=out_dir)
    ood_path, ood_criteria = _build_ood_set(raw_df=raw_df, out_dir=out_dir)

    manifest = {
        "raw_file": str(raw_file),
        "hashfrag_indices": str(hashfrag_dir / "test_indices.npy"),
        "in_distribution_file": str(in_dist_path),
        "snv_pairs_file": str(snv_path),
        "ood_file": str(ood_path),
        "counts": {
            "in_distribution": int(len(pd.read_csv(in_dist_path, sep="\t"))),
            "snv_pairs": int(len(pd.read_csv(snv_path, sep="\t"))),
            "ood": int(len(pd.read_csv(ood_path, sep="\t"))),
        },
        "ood_criteria": ood_criteria,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Created K562 test sets:")
    print(f"- {in_dist_path}")
    print(f"- {snv_path}")
    print(f"- {ood_path}")
    print(f"- {out_dir / 'manifest.json'}")
    print("Counts:", manifest["counts"])


if __name__ == "__main__":
    main()
