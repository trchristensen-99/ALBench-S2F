#!/usr/bin/env python
"""Evaluate Malinois using the boda2 tutorial (load_malinois_model.ipynb) protocol.

Uses boda.common.utils.load_model() for the official artifact (tar.gz or gs://),
FlankBuilder to pad 200-mer sequences to (4, 600), and runs on the same test sets
as our AlphaGenome evals for comparable metrics.

Ref: https://github.com/sjgosai/boda2/blob/main/tutorials/load_malinois_model.ipynb
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boda_dir",
        type=str,
        required=True,
        help="Path to the boda2 repository (for boda.common.utils and constants).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to Malinois artifact: tar.gz (or gs:// URL), or unpacked dir containing "
            "artifacts/torch_checkpoint.pt (or torch_checkpoint.pt)."
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/k562",
        help="Path to K562 data (for chrom test split).",
    )
    parser.add_argument(
        "--test_tsv_dir",
        type=str,
        default=None,
        help="If set, also evaluate on HashFrag TSVs. Default: use only original K562 chrom test (chr 7, 13) from DATA-Table_S2.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/malinois_eval_boda2_tutorial/result.json",
        help="Path to save JSON results.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def _ensure_boda_in_path(boda_dir: str) -> None:
    boda_dir = os.path.abspath(boda_dir)
    if boda_dir not in sys.path:
        sys.path.insert(0, boda_dir)


def _stub_boda_heavy_subpackages() -> None:
    """Stub boda.data, boda.generator, boda.graph so 'import boda.model' doesn't pull in lightning/imageio."""
    import types

    for name in ("boda.data", "boda.generator", "boda.graph"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _load_model_boda2_tutorial(boda_dir: str, model_path: str, device: torch.device):
    """Load Malinois (tutorial style) and FlankBuilder; returns (model, flank_builder)."""
    _stub_boda_heavy_subpackages()
    _ensure_boda_in_path(boda_dir)

    path = model_path.strip()
    if path.endswith(".tar.gz") or path.startswith("gs://"):
        with tempfile.TemporaryDirectory(prefix="malinois_artifact_") as tmpdir:
            if path.startswith("gs://"):
                import subprocess

                subprocess.run(["gsutil", "cp", path, tmpdir], check=True)
                path = os.path.join(tmpdir, os.listdir(tmpdir)[0])
            if path.endswith(".tar.gz"):
                shutil.unpack_archive(path, tmpdir)
            artifacts_dir = os.path.join(tmpdir, "artifacts")
            if not os.path.isdir(artifacts_dir):
                artifacts_dir = tmpdir
            model = _model_fn(artifacts_dir)
            flank_builder = _build_flank_builder(boda_dir, artifacts_dir, device)
            model.to(device)
            model.eval()
            return model, flank_builder
    else:
        artifacts_dir = path
        if not os.path.isfile(os.path.join(artifacts_dir, "torch_checkpoint.pt")):
            artifacts_dir = os.path.join(artifacts_dir, "artifacts")
        model = _model_fn(artifacts_dir)
        flank_builder = _build_flank_builder(boda_dir, artifacts_dir, device)
        model.to(device)
        model.eval()
        return model, flank_builder


def _model_fn(model_dir: str):
    """Load model from unpacked artifact dir (mirrors boda.common.utils.model_fn)."""
    import torch

    _model = __import__("boda.model", fromlist=["model"])
    ckpt_path = os.path.join(model_dir, "torch_checkpoint.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_module = getattr(_model, checkpoint["model_module"])
    model = model_module(**vars(checkpoint["model_hparams"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _build_flank_builder(boda_dir: str, artifact_dir: str, device: torch.device):
    """Build FlankBuilder from artifact input_len and boda MPRA constants (notebook style)."""
    _stub_boda_heavy_subpackages()
    _ensure_boda_in_path(boda_dir)
    import boda.common.constants as constants
    import boda.common.utils as boda_utils

    ckpt_path = os.path.join(artifact_dir, "torch_checkpoint.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(artifact_dir, "..", "torch_checkpoint.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_len = int(checkpoint["model_hparams"].input_len)

    left_pad_len = (input_len - 200) // 2
    right_pad_len = (input_len - 200) - left_pad_len

    left_flank = boda_utils.dna2tensor(constants.MPRA_UPSTREAM[-left_pad_len:]).unsqueeze(0)
    right_flank = boda_utils.dna2tensor(constants.MPRA_DOWNSTREAM[:right_pad_len]).unsqueeze(0)

    flank_builder = boda_utils.FlankBuilder(left_flank=left_flank, right_flank=right_flank)
    flank_builder.to(device)
    return flank_builder


def _encode_200(seq_str: str, boda_dir: str) -> torch.Tensor:
    """One-hot encode sequence to (4, 200). Center-crop or center-pad to 200 bp."""
    _ensure_boda_in_path(boda_dir)
    L = len(seq_str)
    if L > 200:
        start = (L - 200) // 2
        seq_str = seq_str[start : start + 200]
    elif L < 200:
        left = (200 - L) // 2
        seq_str = "N" * left + seq_str + "N" * (200 - L - left)
    try:
        import boda.common.utils as boda_utils

        t = boda_utils.dna2tensor(seq_str)
    except Exception:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.zeros((4, 200), dtype=np.float32)
        for i, c in enumerate(seq_str):
            if c in mapping:
                arr[mapping[c], i] = 1.0
        return torch.tensor(arr, dtype=torch.float32)
    if t.shape[-1] != 200:
        t = (
            torch.nn.functional.pad(t, (0, 200 - t.shape[-1]))
            if t.shape[-1] < 200
            else t[..., :200]
        )
    return t


def _predict_k562_batched(
    model, flank_builder, seqs_str: list, batch_size: int, device: torch.device, boda_dir: str
):
    """Run model on 200-mer sequences (padded to 600 via FlankBuilder); return K562 track."""
    preds = []
    for i in range(0, len(seqs_str), batch_size):
        batch_str = seqs_str[i : i + batch_size]
        batch_4_200 = torch.stack([_encode_200(s, boda_dir) for s in batch_str]).to(device)
        with torch.no_grad():
            prepped = flank_builder(batch_4_200)
            out = model(prepped)
        preds.append(out[:, 0].cpu().numpy())
    return np.concatenate(preds)


def _eval_chrom_test(
    model, flank_builder, data_path: str, batch_size: int, device: torch.device, boda_dir: str
):
    """Chromosome-based test split (chr 7, 13) using K562FullDataset. Sequences are 600 bp (Boda-padded); use middle 200 bp for FlankBuilder."""
    from albench.data.k562_full import K562FullDataset

    ds = K562FullDataset(data_path, split="test")
    seqs = [ds.sequences[j] for j in range(len(ds))]
    labels = ds.labels
    seqs_200 = []
    for s in seqs:
        if len(s) == 600:
            s = s[200:400]
        elif len(s) != 200:
            if len(s) > 200:
                start = (len(s) - 200) // 2
                s = s[start : start + 200]
            else:
                s = s.center(200, "N")
        seqs_200.append(s)
    seqs = seqs_200
    preds = _predict_k562_batched(model, flank_builder, seqs, batch_size, device, boda_dir)
    return {
        "pearson_r": float(pearsonr(labels, preds)[0]),
        "spearman_r": float(spearmanr(labels, preds)[0]),
        "mse": float(np.mean((np.array(labels) - preds) ** 2)),
        "n": int(len(preds)),
    }


def _eval_hashfrag(
    model, flank_builder, test_tsv_dir: str, batch_size: int, device: torch.device, boda_dir: str
):
    """HashFrag ID / SNV_abs / SNV_delta / OOD (same as eval_ag / eval_malinois_baseline)."""
    d = Path(test_tsv_dir)
    out = {}

    in_df = pd.read_csv(d / "test_in_distribution_hashfrag.tsv", sep="\t")
    pred = _predict_k562_batched(
        model, flank_builder, in_df["sequence"].astype(str).tolist(), batch_size, device, boda_dir
    )
    out["in_distribution"] = {
        "pearson_r": float(pearsonr(pred, in_df["K562_log2FC"].to_numpy())[0]),
        "n": len(pred),
    }

    snv_df = pd.read_csv(d / "test_snv_pairs_hashfrag.tsv", sep="\t")
    ref_pred = _predict_k562_batched(
        model,
        flank_builder,
        snv_df["sequence_ref"].astype(str).tolist(),
        batch_size,
        device,
        boda_dir,
    )
    alt_pred = _predict_k562_batched(
        model,
        flank_builder,
        snv_df["sequence_alt"].astype(str).tolist(),
        batch_size,
        device,
        boda_dir,
    )
    out["snv_abs"] = {
        "pearson_r": float(
            pearsonr(
                np.concatenate([ref_pred, alt_pred]),
                np.concatenate(
                    [snv_df["K562_log2FC_ref"].to_numpy(), snv_df["K562_log2FC_alt"].to_numpy()]
                ),
            )[0]
        ),
        "n": len(ref_pred) * 2,
    }
    out["snv_delta"] = {
        "pearson_r": float(pearsonr(alt_pred - ref_pred, snv_df["delta_log2FC"].to_numpy())[0]),
        "n": len(ref_pred),
    }

    ood_df = pd.read_csv(d / "test_ood_cre.tsv", sep="\t")
    ood_pred = _predict_k562_batched(
        model, flank_builder, ood_df["sequence"].astype(str).tolist(), batch_size, device, boda_dir
    )
    out["ood"] = {
        "pearson_r": float(pearsonr(ood_pred, ood_df["K562_log2FC"].to_numpy())[0]),
        "n": len(ood_pred),
    }

    return out


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Malinois (boda2 tutorial protocol) ...")
    model, flank_builder = _load_model_boda2_tutorial(args.boda_dir, args.model_path, device)

    results = {}

    print("Evaluating on chromosome test split (chr 7, 13) ...")
    results["chrom_test"] = _eval_chrom_test(
        model, flank_builder, args.data_path, args.batch_size, device, args.boda_dir
    )
    print(f"  Pearson R: {results['chrom_test']['pearson_r']:.4f}")
    print(f"  Spearman R: {results['chrom_test']['spearman_r']:.4f}")
    print(f"  MSE: {results['chrom_test']['mse']:.4f}")

    if args.test_tsv_dir:
        print(f"Evaluating on HashFrag test sets from {args.test_tsv_dir} ...")
        results["hashfrag"] = _eval_hashfrag(
            model, flank_builder, args.test_tsv_dir, args.batch_size, device, args.boda_dir
        )
        for k, v in results["hashfrag"].items():
            print(f"  {k}: Pearson R = {v['pearson_r']:.4f}  (n={v['n']})")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
