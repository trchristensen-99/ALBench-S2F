#!/bin/bash
#SBATCH --job-name=borzoi_diag3
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

uv run --no-sync python -c "
import torch, torch.nn.functional as F, numpy as np, sys
sys.path.insert(0, '.')
print(f'PyTorch: {torch.__version__}', flush=True)

from data.k562 import K562Dataset
from data.k562_full import MPRA_DOWNSTREAM, MPRA_UPSTREAM
from data.utils import one_hot_encode

# Load model with our fixes
from experiments.train_foundation_stage2 import (
    _load_borzoi, _forward_borzoi, _add_flanks, _rc_onehot, MLPHead
)

device = torch.device('cuda')
model, embed_dim = _load_borzoi()
model.to(device).eval()
for p in model.parameters():
    p.requires_grad = False
print(f'Model loaded, embed_dim={embed_dim}', flush=True)

# Check positions
for i, blk in enumerate(model.transformer):
    attn = blk[0].fn[1]
    pos = attn.positions
    print(f'Block {i}: positions shape={pos.shape} nan={torch.isnan(pos).any().item()} device={pos.device}')

# Load a few val sequences
ds = K562Dataset(data_path='data/k562', split='val')
print(f'Val set: {len(ds)} sequences', flush=True)

# Test embeddings on first 16 sequences
ohs = []
for i in range(16):
    seq_5ch, label = ds[i]
    oh_4ch = np.asarray(seq_5ch)[:4]
    oh_600 = _add_flanks(oh_4ch)
    ohs.append(oh_600)

oh_batch = torch.from_numpy(np.stack(ohs)).float().to(device)  # (16, 4, 600)
print(f'Input batch: shape={oh_batch.shape}', flush=True)
print(f'Input unique per-sample: {[oh_batch[i].sum().item() for i in range(4)]}', flush=True)

with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
    emb = _forward_borzoi(model, oh_batch)  # (16, 1536)

print(f'Embeddings: shape={emb.shape} dtype={emb.dtype}', flush=True)
print(f'Emb nan: {torch.isnan(emb).any().item()}', flush=True)
print(f'Emb range: [{emb.float().min():.4f}, {emb.float().max():.4f}]', flush=True)
print(f'Emb std across samples: {emb.float().std(dim=0).mean():.6f}', flush=True)
print(f'Emb std across features: {emb.float().std(dim=1).mean():.6f}', flush=True)

# Check if embeddings are different for different sequences
for i in range(4):
    for j in range(i+1, 4):
        diff = (emb[i] - emb[j]).float().abs().mean().item()
        print(f'  diff(seq{i}, seq{j}) = {diff:.6f}')

# Also test with torch.no_grad + float32 (no autocast)
with torch.no_grad():
    emb_f32 = _forward_borzoi(model, oh_batch)
print(f'Float32 emb range: [{emb_f32.float().min():.4f}, {emb_f32.float().max():.4f}]')
print(f'Float32 emb std across samples: {emb_f32.float().std(dim=0).mean():.6f}')

# Test step by step to see where variance is lost
with torch.no_grad():
    pad_total = 196608 - 600
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = F.pad(oh_batch, (pad_left, pad_right), value=0.0)

    x = model.conv_dna(padded)
    print(f'After conv_dna: shape={x.shape} std_across_samples={x.std(dim=0).mean():.6f}')

    x_unet0 = model.res_tower(x)
    print(f'After res_tower: shape={x_unet0.shape} std_across_samples={x_unet0.std(dim=0).mean():.6f}')

    x_unet1 = model.unet1(x_unet0)
    print(f'After unet1: shape={x_unet1.shape} std_across_samples={x_unet1.std(dim=0).mean():.6f}')

    x = model._max_pool(x_unet1)
    print(f'After max_pool: shape={x.shape} std_across_samples={x.std(dim=0).mean():.6f}')

    # Transformer
    x_trans = x.permute(0, 2, 1)
    print(f'Transformer input: shape={x_trans.shape} std_across_samples={x_trans.std(dim=0).mean():.6f}')

    for i, blk in enumerate(model.transformer):
        x_trans = blk(x_trans)
        is_nan = torch.isnan(x_trans).any().item()
        std_samples = x_trans.float().std(dim=0).mean().item()
        print(f'Block {i}: nan={is_nan} range=[{x_trans.float().min():.2f}, {x_trans.float().max():.2f}] std_across_samples={std_samples:.6f}')
        if is_nan:
            print('  NaN detected, stopping')
            break

    # Check if the PATCHED forward is being called
    print()
    print('Checking monkey-patch...')
    import inspect
    attn = model.transformer[0][0].fn[1]
    src = inspect.getsource(attn.forward)
    has_slice = 'pos_center' in src
    print(f'Monkey-patch active: {has_slice}')
    print(f'Forward source first 200 chars: {src[:200]}')

print('DONE', flush=True)
"
