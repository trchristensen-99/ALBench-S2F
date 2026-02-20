#!/usr/bin/env python3
"""Create standardized K562 evaluation test sets from available benchmark data.

Outputs under ``data/k562/test_sets``:
- ``test_in_distribution_hashfrag.tsv``: HashFrag test split (expression prediction).
- ``test_snv_pairs_hashfrag.tsv``: ref/alt SNV pairs aligned to HashFrag test loci.
- ``test_ood_cre.tsv``: out-of-domain proxy set from the CRE library.
- ``manifest.json``: criteria and counts used to build each file.

Notes:
- The public Gosai K562 file in this repo may not include ``target_cell`` / ``origin``
  columns. In that case, OOD is built from ``data_project == 'CRE'`` as proxy.
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
    parsed = df.copy()
    parsed["chr_parsed"] = id_parts[0]
    parsed["pos_parsed"] = id_parts[1]
    parsed["ref_parsed"] = id_parts[2]
    parsed["alt_parsed"] = id_parts[3]
    parsed["allele_parsed"] = id_parts[4]
    parsed["variant_key"] = (
        parsed["chr_parsed"]
        + ":"
        + parsed["pos_parsed"]
        + ":"
        + parsed["ref_parsed"]
        + ":"
        + parsed["alt_parsed"]
    )
    return parsed


def _filtered_reference_like(df: pd.DataFrame) -> pd.DataFrame:
    parsed = _parse_ids(df)
    is_reference = parsed["allele_parsed"] == "R"
    is_non_variant = (parsed["ref_parsed"] == "NA") & (parsed["alt_parsed"] == "NA")
    keep = parsed[is_reference | is_non_variant].copy()

    keep["seq_len"] = keep["sequence"].astype(str).str.len()
    keep = keep[keep["seq_len"] >= 198].copy()
    keep = keep.drop(columns=["seq_len"])
    return keep.reset_index(drop=True)


def _build_in_distribution_hashfrag(
    filtered_df: pd.DataFrame, hashfrag_dir: Path, out_dir: Path
) -> tuple[Path, pd.DataFrame]:
    test_idx_path = hashfrag_dir / "test_indices.npy"
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Missing HashFrag test indices: {test_idx_path}")

    test_idx = np.load(test_idx_path)
    in_dist = filtered_df.iloc[test_idx].copy().reset_index(drop=True)
    out_path = out_dir / "test_in_distribution_hashfrag.tsv"
    in_dist.to_csv(out_path, sep="\t", index=False)
    return out_path, in_dist


def _build_snv_pairs_from_hashfrag_test(
    raw_df: pd.DataFrame, in_dist_df: pd.DataFrame, out_dir: Path
) -> Path:
    raw = _parse_ids(raw_df)

    # Reference loci in the hashfrag test set that are true SNVs.
    ref_loci = in_dist_df[
        (in_dist_df["allele_parsed"] == "R")
        & (in_dist_df["ref_parsed"] != "NA")
        & (in_dist_df["alt_parsed"] != "NA")
    ]["variant_key"].drop_duplicates()

    if ref_loci.empty:
        snv_pairs = pd.DataFrame(
            columns=[
                "variant_key",
                "IDs_ref",
                "sequence_ref",
                "K562_log2FC_ref",
                "IDs_alt",
                "sequence_alt",
                "K562_log2FC_alt",
                "delta_log2FC",
            ]
        )
    else:
        ref_rows = raw[(raw["allele_parsed"] == "R") & (raw["variant_key"].isin(ref_loci))].copy()
        alt_rows = raw[(raw["allele_parsed"] == "A") & (raw["variant_key"].isin(ref_loci))].copy()

        ref_rows = ref_rows.drop_duplicates(subset=["variant_key"])
        alt_rows = alt_rows.drop_duplicates(subset=["variant_key"])

        snv_pairs = pd.merge(
            ref_rows[["variant_key", "IDs", "sequence", "K562_log2FC", "K562_lfcSE"]],
            alt_rows[["variant_key", "IDs", "sequence", "K562_log2FC", "K562_lfcSE"]],
            on="variant_key",
            how="inner",
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

    in_dist_path, in_dist_df = _build_in_distribution_hashfrag(
        filtered_df=filtered_df,
        hashfrag_dir=hashfrag_dir,
        out_dir=out_dir,
    )
    snv_path = _build_snv_pairs_from_hashfrag_test(
        raw_df=raw_df, in_dist_df=in_dist_df, out_dir=out_dir
    )
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
