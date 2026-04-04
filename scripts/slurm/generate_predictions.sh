#!/bin/bash
# Generate predictions.npz for ALL existing models with checkpoints.
# Re-evaluates saved checkpoints and saves prediction arrays for scatter plots.
#
# Covers:
#   - chr_split/ (ref-only): DREAM-RNN, DREAM-CNN, Malinois
#   - bar_final/ (ref+alt): all models with best_model.pt
#   - Foundation models: Enformer S1/S2
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/generate_predictions.sh
#
#SBATCH --job-name=gen_preds
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Generate predictions for scatter plots — $(date) ==="

# ── From-scratch models (DREAM-RNN, DREAM-CNN, LegNet) ──────────────
# These have checkpoints at best_model.pt, re-evaluate via exp1_1_scaling.py
# with --save-predictions --eval-only (if supported) or re-train (fast if checkpoint exists)

# For chr_split models, we re-run with --save-predictions
# The training will use the full data but skip epochs since checkpoint exists
# Actually exp1_1_scaling.py doesn't have --eval-only. The simplest approach
# is a Python script that loads checkpoints and runs evaluation.

echo "=== Generating predictions via Python ==="

uv run --no-sync python << 'PYEOF'
import json
import glob
import os
import sys
import numpy as np
import torch
from pathlib import Path

# ── Malinois predictions ──────────────────────────────────────────────
print("\n=== Malinois predictions ===")
sys.path.insert(0, str(Path.cwd()))

from experiments.train_malinois_k562 import (
    BassetBranched, K562MalinoisDataset, K562Dataset,
    _predict_sequences, _evaluate_chr_split_test, evaluate_test_sets,
    CELL_LINE_LABEL_COLS, _LEFT_FLANK, _RIGHT_FLANK,
)
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_malinois_preds(result_dir, n_outputs=1, chr_split=True, include_alt=True, cell_type_idx=None):
    """Load a Malinois checkpoint and save predictions."""
    ckpt_path = result_dir / "best_model.pt"
    pred_path = result_dir / "predictions.npz"
    if pred_path.exists():
        print(f"  {result_dir}: predictions exist, skipping")
        return
    if not ckpt_path.exists():
        print(f"  {result_dir}: no checkpoint, skipping")
        return

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    # Detect architecture from state dict
    has_linear2 = any("linear2" in k for k in sd.keys())
    n_lin = 2 if has_linear2 else 1
    # Detect branched layers
    branched_layers = max(
        (int(k.split("branched_layer_")[1].split(".")[0]) for k in sd if "branched_layer_" in k),
        default=1
    )
    # Detect channels
    if "branched.branched_layer_1.weight" in sd:
        branched_ch = sd["branched.branched_layer_1.weight"].shape[-1]
    else:
        branched_ch = 250

    model = BassetBranched(
        input_len=600, n_outputs=n_outputs,
        n_linear_layers=n_lin, branched_channels=branched_ch,
        n_branched_layers=branched_layers,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    cfg = {"use_reverse_complement": True, "cell_line": "k562"}
    pred_arrays = {}

    if chr_split:
        for ct_name, ct_idx_val in [("k562", 0), ("hepg2", 1), ("sknsh", 2)]:
            if n_outputs == 1 and ct_name != "k562":
                continue
            ct_cfg = {**cfg, "cell_line": ct_name}
            cti = ct_idx_val if n_outputs > 1 else None
            label_col = CELL_LINE_LABEL_COLS.get(ct_name, "K562_log2FC")
            try:
                test_ds = K562MalinoisDataset(K562Dataset(
                    data_path="data/k562", split="test", label_column=label_col,
                    use_hashfrag=False, use_chromosome_fallback=True,
                    include_alt_alleles=include_alt,
                ))
                metrics, preds = _evaluate_chr_split_test(model, device, test_ds, ct_cfg, cell_type_idx=cti)
                suffix = f"_{ct_name}" if n_outputs > 1 else ""
                for k, v in preds.items():
                    pred_arrays[f"{k}{suffix}"] = v
                id_p = metrics.get("in_distribution", {}).get("pearson_r", 0)
                print(f"  {ct_name} in_dist={id_p:.4f}")
            except Exception as e:
                print(f"  {ct_name} error: {e}")
    else:
        try:
            test_dir = Path("data/k562/test_sets")
            metrics, preds = evaluate_test_sets(model, device, test_dir, cfg, cell_type_idx=cell_type_idx)
            pred_arrays = preds
        except Exception as e:
            print(f"  hashfrag error: {e}")

    if pred_arrays:
        np.savez_compressed(str(pred_path), **pred_arrays)
        print(f"  Saved {len(pred_arrays)} arrays to {pred_path}")

# Process all Malinois results
for pattern in [
    "outputs/bar_final/*/malinois/seed_*/",
    "outputs/bar_final/*/malinois_paper/seed_*/",
    "outputs/bar_final/*/malinois_paper_nopretrain/seed_*/",
    "outputs/chr_split/*/malinois/seed_*/seed_*/",
]:
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        if (d / "result.json").exists() or (d / "best_model.pt").exists():
            # Detect n_outputs
            n_out = 1
            if "malinois_paper" in str(d):
                n_out = 3
            # Detect split type
            is_chr = "bar_final" in str(d) or "chr_split" in str(d)
            incl_alt = "bar_final" in str(d)
            print(f"\n  Processing: {d} (n_out={n_out})")
            gen_malinois_preds(d, n_outputs=n_out, chr_split=is_chr, include_alt=incl_alt)

# ── Foundation model predictions ──────────────────────────────────────
print("\n=== Foundation model predictions ===")
from experiments.train_foundation_cached import evaluate_test_sets_cached, MLPHead

for pattern in [
    "outputs/chr_split/*/enformer_s1/seed_*/seed_*/",
    "outputs/enformer_k562_stage2_final/*/run_*/",
    "outputs/enformer_hepg2_stage2/seed_*/",
    "outputs/enformer_sknsh_stage2/seed_*/",
    "outputs/bar_final/*/enformer_s1_v2/seed_*/seed_*/",
]:
    for d in sorted(glob.glob(pattern)):
        d = Path(d)
        pred_path = d / "predictions.npz"
        ckpt_path = d / "best_model.pt"
        if pred_path.exists():
            print(f"  {d}: predictions exist")
            continue
        if not ckpt_path.exists():
            continue
        # Find the corresponding cache dir
        result_file = d / "result.json"
        if not result_file.exists():
            continue
        try:
            result = json.load(open(result_file))
            cfg_data = result.get("config", {})
            cache_dir = cfg_data.get("cache_dir", "")
            embed_dim = int(cfg_data.get("embed_dim", 3072))
            cell_line = cfg_data.get("cell_line", "k562")
            chr_split_flag = cfg_data.get("chr_split", "False")

            if not cache_dir or not Path(cache_dir).exists():
                print(f"  {d}: cache dir missing ({cache_dir})")
                continue

            # Load model
            model = MLPHead(embed_dim=embed_dim).to(device)
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            metrics, preds = evaluate_test_sets_cached(
                model, device, cache_dir, cell_line,
                chr_split=(chr_split_flag in ("True", True)),
            )
            if preds:
                np.savez_compressed(str(pred_path), **preds)
                print(f"  Saved {len(preds)} arrays to {pred_path}")
        except Exception as e:
            print(f"  {d}: error: {e}")

print("\n=== Done generating predictions — " + str(os.popen("date").read().strip()) + " ===")
PYEOF

echo ""
echo "=== Predictions generation DONE — $(date) ==="
