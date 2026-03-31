#!/bin/bash
# Re-evaluate S1 cached models to save test predictions for scatter plots.
# Uses cached embeddings + trained head to predict on test sets and save NPZ.
#
# Array:
#   0: AG all-folds S1 K562 (10 oracle seeds)
#   1: AG fold-1 S1 K562
#   2: Enformer S1 K562 (3 seeds)
#   3: Borzoi S1 K562 (3 seeds)
#   4: NTv3 S1 K562 (3 seeds)
#   5: AG all-folds S1 HepG2
#   6: AG all-folds S1 SKNSH
#
# V100 is fine (cached embeddings, no encoder)
#
#SBATCH --job-name=save_pred
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== save_predictions task=${T} node=${SLURMD_NODENAME} date=$(date) ==="

# Re-run foundation cached training with save_predictions flag
# The training script is fast (~2 min) and will save predictions at the end

case ${T} in
0)
    echo "AG all-folds S1 K562 predictions"
    for SEED_DIR in outputs/ag_hashfrag_oracle_cached/oracle_*/; do
        SEED=$(basename $SEED_DIR)
        if [ ! -f "${SEED_DIR}/test_predictions.npz" ]; then
            echo "  Saving predictions for ${SEED}..."
            # Re-train quickly (will converge in ~1 epoch from existing weights)
            # Actually, we need a lightweight predict-only script
            python3 -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from pathlib import Path
from experiments.train_foundation_cached import MLPHead, ValCachedDataset
from scipy.stats import pearsonr

seed_dir = Path('${SEED_DIR}')
ckpt = torch.load(seed_dir / 'best_model' / 'checkpoint', map_location='cpu')
# This is a JAX checkpoint, not PyTorch - skip AG oracle seeds
print('  AG oracle uses JAX checkpoints - skipping (predictions already in backup)')
" 2>/dev/null || echo "  Skipped ${SEED} (JAX checkpoint)"
        fi
    done
    ;;

2)
    echo "Enformer S1 K562 predictions"
    CACHE="outputs/enformer_k562_cached/embedding_cache"
    for SEED_DIR in outputs/enformer_k562_3seeds/seed_*/; do
        SEED=$(basename $SEED_DIR)
        if [ ! -f "${SEED_DIR}/test_predictions.npz" ]; then
            echo "  Saving predictions for ${SEED}..."
            python3 -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from pathlib import Path
from experiments.train_foundation_cached import MLPHead

seed_dir = Path('${SEED_DIR}')
ckpt_path = seed_dir / 'best_model.pt'
if not ckpt_path.exists():
    print('  No checkpoint found, skipping')
    sys.exit(0)

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
head = MLPHead(3072, 512, 0.1)
head.load_state_dict(ckpt['model_state_dict'])
head.eval()

cache = Path('${CACHE}')
arrays = {}
for prefix in ['test_in_dist', 'test_ood', 'test_snv_ref', 'test_snv_alt']:
    can_path = cache / f'{prefix}_canonical.npy'
    rc_path = cache / f'{prefix}_rc.npy'
    if can_path.exists():
        emb_c = np.load(str(can_path), mmap_mode='r')
        emb_r = np.load(str(rc_path), mmap_mode='r') if rc_path.exists() else emb_c
        emb = (torch.tensor(emb_c, dtype=torch.float32) + torch.tensor(emb_r, dtype=torch.float32)) / 2
        with torch.no_grad():
            preds = head(emb).numpy().reshape(-1)
        arrays[f'{prefix}_pred'] = preds
        print(f'  {prefix}: {len(preds)} predictions')

np.savez_compressed(seed_dir / 'test_predictions.npz', **arrays)
print(f'  Saved to {seed_dir}/test_predictions.npz')
" 2>/dev/null
        fi
    done
    ;;

3)
    echo "Borzoi S1 K562 predictions"
    CACHE="outputs/borzoi_k562_cached/embedding_cache"
    for SEED_DIR in outputs/borzoi_k562_3seeds/seed_*/; do
        SEED=$(basename $SEED_DIR)
        if [ ! -f "${SEED_DIR}/test_predictions.npz" ]; then
            echo "  Saving predictions for ${SEED}..."
            python3 -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from pathlib import Path
from experiments.train_foundation_cached import MLPHead

seed_dir = Path('${SEED_DIR}')
ckpt_path = seed_dir / 'best_model.pt'
if not ckpt_path.exists():
    print('  No checkpoint found, skipping')
    sys.exit(0)

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
head = MLPHead(1536, 512, 0.1)
head.load_state_dict(ckpt['model_state_dict'])
head.eval()

cache = Path('${CACHE}')
arrays = {}
for prefix in ['test_in_dist', 'test_ood', 'test_snv_ref', 'test_snv_alt']:
    can_path = cache / f'{prefix}_canonical.npy'
    rc_path = cache / f'{prefix}_rc.npy'
    if can_path.exists():
        emb_c = np.load(str(can_path), mmap_mode='r')
        emb_r = np.load(str(rc_path), mmap_mode='r') if rc_path.exists() else emb_c
        emb = (torch.tensor(emb_c, dtype=torch.float32) + torch.tensor(emb_r, dtype=torch.float32)) / 2
        with torch.no_grad():
            preds = head(emb).numpy().reshape(-1)
        arrays[f'{prefix}_pred'] = preds
        print(f'  {prefix}: {len(preds)} predictions')

np.savez_compressed(seed_dir / 'test_predictions.npz', **arrays)
print(f'  Saved to {seed_dir}/test_predictions.npz')
" 2>/dev/null
        fi
    done
    ;;

4)
    echo "NTv3 S1 K562 predictions"
    CACHE="outputs/ntv3_post_k562_cached/embedding_cache"
    for SEED_DIR in outputs/ntv3_post_k562_3seeds/seed_*/; do
        SEED=$(basename $SEED_DIR)
        if [ ! -f "${SEED_DIR}/test_predictions.npz" ]; then
            echo "  Saving predictions for ${SEED}..."
            python3 -c "
import sys, json, numpy as np, torch
sys.path.insert(0, '.')
from pathlib import Path
from experiments.train_foundation_cached import MLPHead

seed_dir = Path('${SEED_DIR}')
ckpt_path = seed_dir / 'best_model.pt'
if not ckpt_path.exists():
    print('  No checkpoint found, skipping')
    sys.exit(0)

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
head = MLPHead(1536, 512, 0.1)
head.load_state_dict(ckpt['model_state_dict'])
head.eval()

cache = Path('${CACHE}')
arrays = {}
for prefix in ['test_in_dist', 'test_ood', 'test_snv_ref', 'test_snv_alt']:
    can_path = cache / f'{prefix}_canonical.npy'
    rc_path = cache / f'{prefix}_rc.npy'
    if can_path.exists():
        emb_c = np.load(str(can_path), mmap_mode='r')
        emb_r = np.load(str(rc_path), mmap_mode='r') if rc_path.exists() else emb_c
        emb = (torch.tensor(emb_c, dtype=torch.float32) + torch.tensor(emb_r, dtype=torch.float32)) / 2
        with torch.no_grad():
            preds = head(emb).numpy().reshape(-1)
        arrays[f'{prefix}_pred'] = preds
        print(f'  {prefix}: {len(preds)} predictions')

np.savez_compressed(seed_dir / 'test_predictions.npz', **arrays)
print(f'  Saved to {seed_dir}/test_predictions.npz')
" 2>/dev/null
        fi
    done
    ;;

*)
    echo "Task ${T} not implemented yet"
    ;;
esac

echo "=== Done: $(date) ==="
