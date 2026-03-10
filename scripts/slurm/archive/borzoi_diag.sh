#!/bin/bash
#SBATCH --job-name=borzoi_diag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
from data.k562 import K562Dataset
from data.k562_full import MPRA_UPSTREAM, MPRA_DOWNSTREAM
from data.utils import one_hot_encode

from borzoi_pytorch import Borzoi
if not hasattr(Borzoi, 'all_tied_weights_keys'):
    Borzoi.all_tied_weights_keys = {}
model = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
model.eval().cuda()

ds = K562Dataset(data_path='data/k562', split='train')
_f5 = MPRA_UPSTREAM[-200:]
_f3 = MPRA_DOWNSTREAM[:200]
_map = {'A':0, 'C':1, 'G':2, 'T':3}

def make_oh(seq):
    seq = seq.upper()
    if len(seq)<200:
        pad=200-len(seq)
        seq='N'*(pad//2)+seq+'N'*(pad-pad//2)
    oh = one_hot_encode(seq, add_singleton_channel=False)
    f5 = np.zeros((4,200),dtype=np.float32)
    for i,c in enumerate(_f5):
        if c in _map: f5[_map[c],i]=1.0
    f3 = np.zeros((4,200),dtype=np.float32)
    for i,c in enumerate(_f3):
        if c in _map: f3[_map[c],i]=1.0
    return np.concatenate([f5, oh, f3], axis=1)

seqs = [make_oh(ds.sequences[i]) for i in range(4)]
oh = torch.from_numpy(np.stack(seqs)).float().cuda()
pad_total = 196608 - oh.shape[2]
padded = F.pad(oh, (pad_total//2, pad_total-pad_total//2), value=0.0)
print('Input shape:', padded.shape, flush=True)
print(f'Input stats: min={padded.min():.4f} max={padded.max():.4f} mean={padded.mean():.6f} sum={padded.sum():.1f}', flush=True)
print(f'oh 600bp stats: min={oh.min():.4f} max={oh.max():.4f} sum_per_pos={oh[:,:,:600].sum(dim=1).mean():.4f}', flush=True)

# Quick sanity: test with random input
rand_input = torch.randn(4, 4, 196608, device='cuda')
with torch.no_grad():
    rand_emb = model.get_embs_after_crop(rand_input)
    print(f'random_input: nan={torch.isnan(rand_emb).any().item()} range=[{rand_emb.min():.2f}, {rand_emb.max():.2f}]', flush=True)
del rand_input, rand_emb

# Also test with the padded input under torch.no_grad() explicitly
with torch.no_grad():
    test_emb = model.get_embs_after_crop(padded)
    print(f'real_no_grad_explicit: nan={torch.isnan(test_emb).any().item()} range=[{test_emb.float().min():.2f}, {test_emb.float().max():.2f}]', flush=True)
del test_emb

# Center bins extraction
NBINS = 6144
center = NBINS // 2
half = 10

def run_test(label, requires_grad_fn, autocast_enabled, autocast_dtype=torch.float16):
    for n,p in model.named_parameters():
        p.requires_grad = requires_grad_fn(n)
    ctx = torch.amp.autocast('cuda', dtype=autocast_dtype, enabled=autocast_enabled)
    with ctx:
        emb = model.get_embs_after_crop(padded)
        center_emb = emb[:, :, center-half:center+half].mean(dim=2)
    nan_full = torch.isnan(emb).any().item()
    nan_center = torch.isnan(center_emb).any().item()
    print(f'{label}: full_nan={nan_full} center_nan={nan_center} '
          f'full_range=[{emb.float().min():.2f}, {emb.float().max():.2f}] '
          f'center_range=[{center_emb.float().min():.2f}, {center_emb.float().max():.2f}]',
          flush=True)
    return not nan_center

# All combinations
run_test('no_grad fp32', lambda n: False, False)
run_test('no_grad fp16', lambda n: False, True)
run_test('grad_all fp32', lambda n: n.startswith('transformer.'), False)
run_test('grad_all fp16', lambda n: n.startswith('transformer.'), True)
run_test('grad_last2 fp32', lambda n: n.startswith('transformer.6.') or n.startswith('transformer.7.'), False)
run_test('grad_last2 fp16', lambda n: n.startswith('transformer.6.') or n.startswith('transformer.7.'), True)
run_test('grad_last2 bf16', lambda n: n.startswith('transformer.6.') or n.startswith('transformer.7.'), True, torch.bfloat16)
run_test('grad_9_10 fp32', lambda n: n.startswith('transformer.9.') or n.startswith('transformer.10.'), False)
run_test('grad_9_10 fp16', lambda n: n.startswith('transformer.9.') or n.startswith('transformer.10.'), True)
print('DONE', flush=True)
"
