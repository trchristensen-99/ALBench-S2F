#!/bin/bash
# Generate predictions.npz for ALL from-scratch models with checkpoints.
# Loads each best_model.pt, evaluates on chr-split test set, saves predictions.
#
#SBATCH --job-name=gen_pred3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Generate predictions for all models — $(date) ==="

uv run --no-sync python << 'PYEOF'
import json, glob, os, sys
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, os.getcwd())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helper: evaluate and save predictions ────────────────────────────
def save_preds_for_model(model, model_dir, cell, include_alt, cell_type_idx=None):
    """Evaluate model on chr-split test and save predictions.npz."""
    pred_path = Path(model_dir) / "predictions.npz"
    if pred_path.exists():
        return  # already done

    from experiments.train_malinois_k562 import (
        K562MalinoisDataset, K562Dataset, CELL_LINE_LABEL_COLS,
        _evaluate_chr_split_test,
    )

    label_col = CELL_LINE_LABEL_COLS.get(cell, "K562_log2FC")
    cfg = {"use_reverse_complement": True, "cell_line": cell}

    try:
        test_ds = K562MalinoisDataset(K562Dataset(
            data_path="data/k562", split="test", label_column=label_col,
            use_hashfrag=False, use_chromosome_fallback=True,
            include_alt_alleles=include_alt,
        ))
        metrics, preds = _evaluate_chr_split_test(
            model, device, test_ds, cfg, cell_type_idx=cell_type_idx
        )
        if preds:
            np.savez_compressed(str(pred_path), **preds)
            id_p = metrics.get("in_distribution", {}).get("pearson_r", 0)
            print(f"  Saved {len(preds)} arrays, in_dist={id_p:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

# ── Malinois (single-task, our arch) ─────────────────────────────────
print("\n=== Malinois (our arch) ===")
from models.basset_branched import BassetBranched

for pattern in [
    "outputs/bar_final/*/malinois/seed_*/seed_*/",
    "outputs/chr_split/*/malinois/seed_*/seed_*/",
]:
    for d in sorted(glob.glob(pattern)):
        ckpt = os.path.join(d, "best_model.pt")
        if not os.path.exists(ckpt):
            continue
        if os.path.exists(os.path.join(d, "predictions.npz")):
            continue
        cell = d.split("/")[2]
        print(f"  {d}")
        try:
            state = torch.load(ckpt, map_location="cpu")
            model = BassetBranched(input_len=600, n_outputs=1).to(device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            include_alt = "bar_final" in d
            save_preds_for_model(model, d, cell, include_alt)
        except Exception as e:
            print(f"  Error: {e}")

# ── DREAM-RNN ────────────────────────────────────────────────────────
print("\n=== DREAM-RNN ===")
from models.dream_rnn import create_dream_rnn
from data.utils import one_hot_encode

for pattern in [
    "outputs/bar_final/*/dream_rnn/genomic/*/hp*/seed*/",
    "outputs/chr_split/*/dream_rnn/genomic/*/hp*/seed*/",
]:
    for d in sorted(glob.glob(pattern)):
        ckpt = os.path.join(d, "best_model.pt")
        if not os.path.exists(ckpt):
            continue
        if os.path.exists(os.path.join(d, "predictions.npz")):
            continue
        cell = d.split("/")[2]
        print(f"  {d}")
        try:
            state = torch.load(ckpt, map_location="cpu")
            model = create_dream_rnn(input_channels=5, sequence_length=200, task_mode="k562")
            model.load_state_dict(state["model_state_dict"])
            model.to(device).eval()
            include_alt = "bar_final" in d

            # DREAM-RNN needs custom predict (5-channel input with RC flag)
            from experiments.train_malinois_k562 import (
                K562MalinoisDataset, K562Dataset, CELL_LINE_LABEL_COLS,
            )
            label_col = CELL_LINE_LABEL_COLS.get(cell, "K562_log2FC")
            test_ds = K562Dataset(
                data_path="data/k562", split="test", label_column=label_col,
                use_hashfrag=False, use_chromosome_fallback=True,
                include_alt_alleles=include_alt,
            )
            # Encode with 5 channels
            from torch.utils.data import DataLoader
            loader = DataLoader(test_ds, batch_size=512, shuffle=False)
            preds_list, trues_list = [], []
            model.eval()
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    p = model.predict(x, use_reverse_complement=True)
                    preds_list.append(p.cpu().numpy().reshape(-1))
                    trues_list.append(y.numpy().reshape(-1))
            pred = np.concatenate(preds_list)
            true = np.concatenate(trues_list)
            r = float(pearsonr(pred, true)[0])
            pred_path = os.path.join(d, "predictions.npz")
            np.savez_compressed(pred_path, in_dist_pred=pred, in_dist_true=true)
            print(f"  Saved in_dist predictions, pearson={r:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

# ── LegNet ───────────────────────────────────────────────────────────
print("\n=== LegNet ===")
from models.legnet import LegNet

for pattern in [
    "outputs/bar_final/*/legnet/genomic/*/hp*/seed*/",
    "outputs/chr_split/*/legnet/genomic/*/hp*/seed*/",
]:
    for d in sorted(glob.glob(pattern)):
        ckpt = os.path.join(d, "best_model.pt")
        if not os.path.exists(ckpt):
            continue
        if os.path.exists(os.path.join(d, "predictions.npz")):
            continue
        cell = d.split("/")[2]
        print(f"  {d}")
        try:
            state = torch.load(ckpt, map_location="cpu")
            model = LegNet(in_channels=4, task_mode="k562").to(device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            include_alt = "bar_final" in d

            from experiments.train_malinois_k562 import (
                K562Dataset, CELL_LINE_LABEL_COLS,
            )
            label_col = CELL_LINE_LABEL_COLS.get(cell, "K562_log2FC")
            test_ds = K562Dataset(
                data_path="data/k562", split="test", label_column=label_col,
                use_hashfrag=False, use_chromosome_fallback=True,
                include_alt_alleles=include_alt,
            )
            from torch.utils.data import DataLoader
            loader = DataLoader(test_ds, batch_size=512, shuffle=False)
            preds_list, trues_list = [], []
            with torch.no_grad():
                for x, y in loader:
                    # LegNet uses 4 channels (strip RC flag if present)
                    x = x[:, :4].to(device)
                    p = model(x).cpu().numpy().reshape(-1)
                    preds_list.append(p)
                    trues_list.append(y.numpy().reshape(-1))
            pred = np.concatenate(preds_list)
            true = np.concatenate(trues_list)
            r = float(pearsonr(pred, true)[0])
            pred_path = os.path.join(d, "predictions.npz")
            np.savez_compressed(pred_path, in_dist_pred=pred, in_dist_true=true)
            print(f"  Saved in_dist predictions, pearson={r:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

# ── DREAM-CNN ────────────────────────────────────────────────────────
print("\n=== DREAM-CNN ===")
from models.dream_cnn import create_dream_cnn

for pattern in [
    "outputs/bar_final/*/dream_cnn/genomic/*/hp*/seed*/",
    "outputs/chr_split/*/dream_cnn/genomic/*/hp*/seed*/",
]:
    for d in sorted(glob.glob(pattern)):
        ckpt = os.path.join(d, "best_model.pt")
        if not os.path.exists(ckpt):
            continue
        if os.path.exists(os.path.join(d, "predictions.npz")):
            continue
        cell = d.split("/")[2]
        print(f"  {d}")
        try:
            state = torch.load(ckpt, map_location="cpu")
            model = create_dream_cnn(input_channels=4, sequence_length=200, task_mode="k562")
            model.load_state_dict(state["model_state_dict"])
            model.to(device).eval()
            include_alt = "bar_final" in d

            from experiments.train_malinois_k562 import (
                K562Dataset, CELL_LINE_LABEL_COLS,
            )
            label_col = CELL_LINE_LABEL_COLS.get(cell, "K562_log2FC")
            test_ds = K562Dataset(
                data_path="data/k562", split="test", label_column=label_col,
                use_hashfrag=False, use_chromosome_fallback=True,
                include_alt_alleles=include_alt,
            )
            from torch.utils.data import DataLoader
            loader = DataLoader(test_ds, batch_size=512, shuffle=False)
            preds_list, trues_list = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x[:, :4].to(device)
                    p = model(x).cpu().numpy().reshape(-1)
                    preds_list.append(p)
                    trues_list.append(y.numpy().reshape(-1))
            pred = np.concatenate(preds_list)
            true = np.concatenate(trues_list)
            r = float(pearsonr(pred, true)[0])
            pred_path = os.path.join(d, "predictions.npz")
            np.savez_compressed(pred_path, in_dist_pred=pred, in_dist_true=true)
            print(f"  Saved in_dist predictions, pearson={r:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

print(f"\n=== Done — total predictions.npz: {len(glob.glob('outputs/**/predictions.npz', recursive=True))} ===")
PYEOF

echo "=== Predictions generation DONE — $(date) ==="
