#!/bin/bash
#SBATCH --job-name=borzoi_diag4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

uv run --no-sync python -c "
import torch, torch.nn.functional as F, numpy as np, sys
sys.path.insert(0, '.')
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from experiments.train_foundation_stage2 import (
    _load_borzoi, _forward_borzoi, _collate_eval, MLPHead, _safe_corr
)
from data.k562 import K562Dataset
from torch.utils.data import DataLoader, Subset

device = torch.device('cuda')

# Load model
model, embed_dim = _load_borzoi()
model.to(device).eval()
for p in model.parameters():
    p.requires_grad = False
print(f'Model loaded, embed_dim={embed_dim}', flush=True)

# Find best S1 head
import json
s1_base = Path('outputs/foundation_grid_search/borzoi')
best_dir, best_val = None, -1.0
for d in s1_base.iterdir():
    for rfile in d.glob('seed_*/result.json'):
        r = json.load(open(rfile))
        vp = r.get('best_val_pearson_r', 0)
        if vp > best_val:
            best_val = vp
            best_dir = rfile.parent
print(f'Best S1: {best_dir} (val_pearson={best_val:.4f})', flush=True)

# Load S1 head
head = MLPHead(embed_dim, 512, 0.1)
ckpt = torch.load(best_dir / 'best_model.pt', map_location='cpu', weights_only=True)
head.load_state_dict(ckpt['model_state_dict'])
head.to(device).eval()
print('S1 head loaded', flush=True)

# Load val set
ds = K562Dataset(data_path='data/k562', split='val')
rng = np.random.RandomState(42)
val_indices = rng.choice(len(ds), 2000, replace=False)
ds = Subset(ds, val_indices)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=_collate_eval, pin_memory=True)

# Run validation (FROZEN model, no training)
val_preds, val_trues = [], []
with torch.no_grad():
    for can_batch, rc_batch, labels in loader:
        can_batch = can_batch.to(device, non_blocking=True)
        rc_batch = rc_batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            emb_can = _forward_borzoi(model, can_batch)
            emb_rc = _forward_borzoi(model, rc_batch)
            p_can = head(emb_can)
            p_rc = head(emb_rc)
        avg_pred = ((p_can + p_rc) / 2.0).cpu().float().numpy()
        val_preds.append(avg_pred)
        val_trues.append(labels.numpy())

preds = np.concatenate(val_preds)
trues = np.concatenate(val_trues)
print(f'Preds: min={preds.min():.4f} max={preds.max():.4f} std={preds.std():.6f} mean={preds.mean():.4f}')
print(f'Trues: min={trues.min():.4f} max={trues.max():.4f} std={trues.std():.6f} mean={trues.mean():.4f}')
pr = _safe_corr(preds, trues, pearsonr)
sr = _safe_corr(preds, trues, spearmanr)
print(f'FROZEN model + S1 head: pearson={pr:.4f} spearman={sr:.4f}')

# Also test with S1 cache embeddings for comparison
print()
print('=== Comparing with S1 cached embeddings ===')
cache_can = np.load('outputs/borzoi_k562_cached/embedding_cache/val_canonical.npy')
cache_rc = np.load('outputs/borzoi_k562_cached/embedding_cache/val_rc.npy')
cache_sub_can = cache_can[val_indices]
cache_sub_rc = cache_rc[val_indices]
head.eval()
with torch.no_grad():
    can_t = torch.from_numpy(cache_sub_can.astype(np.float32)).to(device)
    rc_t = torch.from_numpy(cache_sub_rc.astype(np.float32)).to(device)
    p_can_cached = head(can_t).cpu().numpy()
    p_rc_cached = head(rc_t).cpu().numpy()
    preds_cached = (p_can_cached + p_rc_cached) / 2.0
print(f'Cached preds: min={preds_cached.min():.4f} max={preds_cached.max():.4f} std={preds_cached.std():.6f}')
pr_c = _safe_corr(preds_cached, trues, pearsonr)
sr_c = _safe_corr(preds_cached, trues, spearmanr)
print(f'S1 cached embeddings + S1 head: pearson={pr_c:.4f} spearman={sr_c:.4f}')

# Compare live vs cached embeddings
# Collect live embeddings for comparison
live_embs = []
with torch.no_grad():
    for can_batch, rc_batch, labels in loader:
        can_batch = can_batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            emb = _forward_borzoi(model, can_batch)
        live_embs.append(emb.cpu().float().numpy())
live_embs = np.concatenate(live_embs)
print(f'Live embs range: [{live_embs.min():.4f}, {live_embs.max():.4f}] std_samples={live_embs.std(axis=0).mean():.6f}')
print(f'Cache embs range: [{cache_sub_can.min():.4f}, {cache_sub_can.max():.4f}] std_samples={cache_sub_can.std(axis=0).mean():.6f}')
diff = np.abs(live_embs - cache_sub_can.astype(np.float32)).mean()
print(f'Mean abs diff live vs cache: {diff:.6f}')
corr = np.corrcoef(live_embs.flatten()[:10000], cache_sub_can.astype(np.float32).flatten()[:10000])[0,1]
print(f'Correlation live vs cache (first 10K values): {corr:.6f}')

print('DONE', flush=True)
"
