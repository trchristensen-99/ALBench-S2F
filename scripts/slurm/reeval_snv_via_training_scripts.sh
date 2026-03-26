#!/bin/bash
# Re-evaluate SNV for HepG2/SK-N-SH by re-running training scripts in eval-only mode.
#
# For each model checkpoint, re-runs the original training script's evaluation
# function with the cell-specific test_set_dir (data/{cell}/test_sets/) which
# has the correct HepG2_log2FC_alt / SKNSH_log2FC_alt columns.
#
# Array:
#   0-5:  Foundation S1 (Enformer/Borzoi/NTv3 × HepG2/SKNSH)
#   6-7:  Malinois HepG2/SKNSH
#   8-9:  Enformer S2 HepG2/SKNSH
#   10-11: NTv3 S2 HepG2/SKNSH
#
# Usage:
#   sbatch --array=0-11 scripts/slurm/reeval_snv_via_training_scripts.sh
#
#SBATCH --job-name=reeval_snv3
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

# Task definitions: model_type cell encoder_name result_dirs...
case $SLURM_ARRAY_TASK_ID in
    0)  MODEL="foundation_s1"; CELL="hepg2"; ENC="enformer"; DIRS="outputs/enformer_hepg2_cached/seed_0/seed_0 outputs/enformer_hepg2_cached/seed_1/seed_1 outputs/enformer_hepg2_cached/seed_2/seed_2" ;;
    1)  MODEL="foundation_s1"; CELL="sknsh"; ENC="enformer"; DIRS="outputs/enformer_sknsh_cached/seed_0/seed_0 outputs/enformer_sknsh_cached/seed_1/seed_1 outputs/enformer_sknsh_cached/seed_2/seed_2" ;;
    2)  MODEL="foundation_s1"; CELL="hepg2"; ENC="borzoi"; DIRS="outputs/borzoi_hepg2_cached/seed_0/seed_0 outputs/borzoi_hepg2_cached/seed_1/seed_1 outputs/borzoi_hepg2_cached/seed_2/seed_2" ;;
    3)  MODEL="foundation_s1"; CELL="sknsh"; ENC="borzoi"; DIRS="outputs/borzoi_sknsh_cached/seed_0/seed_0 outputs/borzoi_sknsh_cached/seed_1/seed_1 outputs/borzoi_sknsh_cached/seed_2/seed_2" ;;
    4)  MODEL="foundation_s1"; CELL="hepg2"; ENC="ntv3_post"; DIRS="outputs/ntv3_post_hepg2_cached/seed_0/seed_0 outputs/ntv3_post_hepg2_cached/seed_1/seed_1 outputs/ntv3_post_hepg2_cached/seed_2/seed_2" ;;
    5)  MODEL="foundation_s1"; CELL="sknsh"; ENC="ntv3_post"; DIRS="outputs/ntv3_post_sknsh_cached/seed_0/seed_0 outputs/ntv3_post_sknsh_cached/seed_1/seed_1 outputs/ntv3_post_sknsh_cached/seed_2/seed_2" ;;
    6)  MODEL="malinois"; CELL="hepg2"; ENC=""; DIRS="outputs/malinois_hepg2_3seeds/seed_0/seed_0 outputs/malinois_hepg2_3seeds/seed_1/seed_1 outputs/malinois_hepg2_3seeds/seed_2/seed_2" ;;
    7)  MODEL="malinois"; CELL="sknsh"; ENC=""; DIRS="outputs/malinois_sknsh_3seeds/seed_0/seed_0 outputs/malinois_sknsh_3seeds/seed_1/seed_1 outputs/malinois_sknsh_3seeds/seed_2/seed_2" ;;
    8)  MODEL="enformer_s2"; CELL="hepg2"; ENC="enformer"; DIRS="outputs/enformer_hepg2_stage2/seed_0" ;;
    9)  MODEL="enformer_s2"; CELL="sknsh"; ENC="enformer"; DIRS="outputs/enformer_sknsh_stage2/seed_0" ;;
    10) MODEL="ntv3_s2"; CELL="hepg2"; ENC="ntv3_post"; DIRS="outputs/ntv3_post_hepg2_stage2/seed_0" ;;
    11) MODEL="ntv3_s2"; CELL="sknsh"; ENC="ntv3_post"; DIRS="outputs/ntv3_post_sknsh_stage2/seed_0" ;;
esac

echo "=== SNV Re-eval: ${MODEL} ${CELL} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# The key: use each model's eval function with cell-specific test_set_dir
# Foundation S1 and Malinois use train_foundation_cached.py's evaluate
# S2 models use train_foundation_stage2.py's evaluate

for RD in ${DIRS}; do
    echo ""
    echo "--- ${RD} ---"

    uv run --no-sync python3 -c "
import json, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

REPO = Path('.')
rd = Path('${RD}')
cell = '${CELL}'
model_type = '${MODEL}'
enc_name = '${ENC}'

fc_col = {'hepg2': 'HepG2_log2FC', 'sknsh': 'SKNSH_log2FC'}[cell]

# Load cell-specific SNV file
snv_path = REPO / 'data' / cell / 'test_sets' / 'test_snv_pairs_hashfrag.tsv'
snv_df = pd.read_csv(snv_path, sep='\t')
ref_seqs = snv_df['sequence_ref'].tolist()
alt_seqs = snv_df['sequence_alt'].tolist()
alt_col = fc_col + '_alt'
delta_col = 'delta_' + fc_col
alt_true = snv_df[alt_col].to_numpy(dtype=np.float32)
delta_true = snv_df[delta_col].to_numpy(dtype=np.float32)

def safe_corr(x, y, fn):
    m = np.isfinite(x) & np.isfinite(y)
    return float(fn(x[m], y[m])[0]) if m.sum() > 2 else 0.0

def metrics(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    return {'pearson_r': safe_corr(pred, true, pearsonr), 'spearman_r': safe_corr(pred, true, spearmanr),
            'mse': float(np.mean((pred[m]-true[m])**2)), 'n': int(m.sum())}

if model_type in ('foundation_s1', 'enformer_s2', 'ntv3_s2'):
    from scripts.eval_ood_multicell import _load_foundation_s1, predict_foundation_s1

    if model_type == 'foundation_s1':
        head, enc, device = _load_foundation_s1(rd, enc_name)
        if head is None:
            print('    ERROR: Could not load model')
            sys.exit(1)

        def predict_fn(seqs):
            return predict_foundation_s1(head, enc, seqs, device)

    elif model_type in ('enformer_s2', 'ntv3_s2'):
        # S2 models: for now skip (SNV correction is small)
        # The S2 sweep will produce fresh results with correct eval
        print('    SKIP: S2 models will be re-evaluated by sweep jobs')
        sys.exit(0)

elif model_type == 'malinois':
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint properly
    ckpt_path = rd / 'best_model.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    from models.basset_branched import BassetBranched
    model = BassetBranched()
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()

    from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
    f5, f3 = MPRA_UPSTREAM[-200:], MPRA_DOWNSTREAM[:200]
    mapping = {'A':0,'C':1,'G':2,'T':3}

    def predict_fn(seqs, bs=256):
        preds = []
        for i in range(0, len(seqs), bs):
            batch = seqs[i:i+bs]
            enc = []
            for seq in batch:
                full = f5 + seq[:200] + f3
                oh = np.zeros((4,600), dtype=np.float32)
                for j,c in enumerate(full[:600]):
                    if c in mapping: oh[mapping[c],j] = 1.0
                enc.append(oh)
            x = torch.tensor(np.stack(enc), device=device)
            with torch.no_grad():
                preds.append(model(x).cpu().numpy().reshape(-1))
        return np.concatenate(preds)

print('  Predicting on %d ref + %d alt sequences...' % (len(ref_seqs), len(alt_seqs)))
ref_p = predict_fn(ref_seqs)
alt_p = predict_fn(alt_seqs)

snv_abs = metrics(alt_p, alt_true)
snv_delta = metrics(alt_p - ref_p, delta_true)

print('  snv_abs: r=%.4f' % snv_abs['pearson_r'])
print('  snv_delta: r=%.4f' % snv_delta['pearson_r'])

# Patch result JSON
for jn in ['result.json', 'test_metrics.json']:
    jp = rd / jn
    if jp.exists():
        d = json.loads(jp.read_text())
        c = d.get('test_metrics', d)
        c['snv_abs'] = snv_abs
        c['snv_delta'] = snv_delta
        jp.write_text(json.dumps(d, indent=2, default=str) + '\n')
        print('  Patched', jp)
        break
"
done

echo "Done: $(date)"
