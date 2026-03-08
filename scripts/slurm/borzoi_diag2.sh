#!/bin/bash
#SBATCH --job-name=borzoi_diag2
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
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from borzoi_pytorch import Borzoi
if not hasattr(Borzoi, 'all_tied_weights_keys'):
    Borzoi.all_tied_weights_keys = {}
model = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
model.eval().cuda()
for p in model.parameters():
    p.requires_grad = False
print('Model loaded OK')

# Check positions buffer in each attention block
for i, blk in enumerate(model.transformer):
    attn = blk[0].fn[1]  # Residual -> Sequential -> [0]=LN, [1]=Attention
    pos = attn.positions
    print(f'Block {i} positions: shape={pos.shape} nan={torch.isnan(pos).any().item()} device={pos.device}')

# ============================================================
# Test 1: Standard 524288bp input (should match N=4096)
# ============================================================
print()
print('=== Test 1: Standard 524288bp input ===')
x_524k = torch.randn(1, 4, 524288, device='cuda')
with torch.no_grad():
    try:
        emb = model.get_embs_after_crop(x_524k)
        print(f'524288bp: nan={torch.isnan(emb).any().item()} range=[{emb.min():.2f}, {emb.max():.2f}]')
    except Exception as e:
        print(f'524288bp: ERROR {e}')
del x_524k
torch.cuda.empty_cache()

# ============================================================
# Test 2: 196608bp input (N=1536, mismatched positions)
# ============================================================
print()
print('=== Test 2: 196608bp input ===')
x_196k = torch.randn(1, 4, 196608, device='cuda')
with torch.no_grad():
    try:
        emb = model.get_embs_after_crop(x_196k)
        print(f'196608bp: nan={torch.isnan(emb).any().item()} range=[{emb.min():.2f}, {emb.max():.2f}]')
    except Exception as e:
        print(f'196608bp: ERROR {e}')

# ============================================================
# Test 3: Manual transformer forward (block by block) for 196608bp
# ============================================================
print()
print('=== Test 3: Manual block-by-block for 196608bp ===')
with torch.no_grad():
    x = model.conv_dna(x_196k)
    x_unet0 = model.res_tower(x)
    x_unet1 = model.unet1(x_unet0)
    x = model._max_pool(x_unet1)
    x = x.permute(0, 2, 1)  # (B, N, 1536)
    print(f'Transformer input: shape={x.shape} nan={torch.isnan(x).any().item()}')
    N = x.shape[1]
    print(f'Sequence length N={N}')
    print(f'Expected rel_k size for N={N}: {2*N-1}')
    print(f'Actual positions buffer size: {model.transformer[0][0].fn[1].positions.shape[0]}')

    for i, blk in enumerate(model.transformer):
        x_before = x.clone()
        x = blk(x)
        is_nan = torch.isnan(x).any().item()
        print(f'Block {i}: nan={is_nan} range=[{x.float().min():.2f}, {x.float().max():.2f}]')
        if is_nan:
            print(f'  NaN at block {i}! Investigating attention...')
            # Manually run the attention
            attn_res = blk[0]  # First Residual (attention)
            attn_seq = attn_res.fn  # Sequential: LN, Attention, Dropout
            ln = attn_seq[0]
            attention = attn_seq[1]

            ln_out = ln(x_before)
            print(f'  LN output: nan={torch.isnan(ln_out).any().item()} range=[{ln_out.min():.4f}, {ln_out.max():.4f}]')

            # Manual Q, K, V
            q = attention.to_q(ln_out)
            k = attention.to_k(ln_out)
            v = attention.to_v(ln_out)
            print(f'  Q: nan={torch.isnan(q).any().item()} range=[{q.min():.4f}, {q.max():.4f}]')
            print(f'  K: nan={torch.isnan(k).any().item()} range=[{k.min():.4f}, {k.max():.4f}]')
            print(f'  V: nan={torch.isnan(v).any().item()} range=[{v.min():.4f}, {v.max():.4f}]')

            from einops import rearrange
            h = attention.heads
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
            q2 = q2 * attention.scale

            # Content logits
            from torch import einsum
            content_logits = einsum('b h i d, b h j d -> b h i j', q2 + attention.rel_content_bias, k2)
            print(f'  Content logits: nan={torch.isnan(content_logits).any().item()} range=[{content_logits.min():.4f}, {content_logits.max():.4f}]')

            # Relative position logits
            positions = attention.pos_dropout(attention.positions)
            print(f'  Positions: nan={torch.isnan(positions).any().item()} shape={positions.shape}')
            rel_k = attention.to_rel_k(positions)
            rel_k = rearrange(rel_k, 'n (h d) -> h n d', h=h)
            print(f'  rel_k: nan={torch.isnan(rel_k).any().item()} shape={rel_k.shape}')

            # Test fast_relative_shift
            from borzoi_pytorch.pytorch_borzoi_transformer import fast_relative_shift
            try:
                rel_logits = fast_relative_shift(q2 + attention.rel_pos_bias, rel_k)
                print(f'  Rel logits (vmapped): nan={torch.isnan(rel_logits).any().item()} shape={rel_logits.shape} range=[{rel_logits.float().min():.4f}, {rel_logits.float().max():.4f}]')
            except Exception as e:
                print(f'  Rel logits (vmapped): ERROR {e}')

            # Manual relative shift (no vmap, loop over batch and heads)
            print('  Testing manual rel shift (no vmap)...')
            q_shifted = q2 + attention.rel_pos_bias  # (B, H, N, d)
            B_sz, H_sz, N_sz, d_sz = q_shifted.shape
            rel_logits_manual = torch.zeros(B_sz, H_sz, N_sz, N_sz, device=q_shifted.device)
            for b_idx in range(B_sz):
                for h_idx in range(H_sz):
                    a_single = q_shifted[b_idx, h_idx]  # (N, d)
                    b_single = rel_k[h_idx]  # (M, d)
                    raw = einsum('i d, j d -> i j', a_single, b_single)  # (N, M)
                    # Correct relative shift: for each query i attending to key j,
                    # we want the relative position embedding at distance (j - i)
                    # In the raw tensor, distance d maps to column (M//2 + d)
                    # So for query i, key j: column = M//2 + (j - i)
                    M = b_single.shape[0]
                    center = M // 2  # = 4095
                    for qi in range(min(N_sz, 3)):  # just test first 3
                        for kj in range(min(N_sz, 3)):
                            col_idx = center + (kj - qi)
                            if 0 <= col_idx < M:
                                rel_logits_manual[b_idx, h_idx, qi, kj] = raw[qi, col_idx]
                    break  # just first head
                break  # just first batch

            # Check if the manual result has NaN
            print(f'  Manual rel logits (first 3x3): nan={torch.isnan(rel_logits_manual[:1,:1,:3,:3]).any().item()}')
            print(f'  Values: {rel_logits_manual[0,0,:3,:3]}')

            # Now test: what does fast_relative_shift produce vs correct?
            a_single = q_shifted[0, 0]  # (N, d)
            b_single = rel_k[0]  # (M, d)
            raw_single = einsum('i d, j d -> i j', a_single, b_single)  # (N, M)
            flat = raw_single.flatten()
            print(f'  raw einsum: shape={raw_single.shape} nan={torch.isnan(raw_single).any().item()}')

            # Apply the stride trick manually
            try:
                strided = flat.as_strided(
                    size=(a_single.shape[0], a_single.shape[0]),
                    stride=((a_single.shape[0]-1)*2, 1),
                    storage_offset=a_single.shape[0] - 1
                )
                print(f'  as_strided result: nan={torch.isnan(strided).any().item()} shape={strided.shape}')
                print(f'  as_strided[0,:5] = {strided[0,:5]}')
                print(f'  as_strided[-1,-5:] = {strided[-1,-5:]}')
            except Exception as e:
                print(f'  as_strided: ERROR {e}')

            # Check if the stride goes out of bounds conceptually
            max_idx = (a_single.shape[0] - 1) + (a_single.shape[0] - 1) * ((a_single.shape[0]-1)*2) + (a_single.shape[0] - 1)
            print(f'  Max strided index: {max_idx}, total elements: {flat.shape[0]}, OOB: {max_idx >= flat.shape[0]}')

            break  # stop after first NaN block
del x_196k
torch.cuda.empty_cache()

# ============================================================
# Test 4: Fix the positions buffer to match N=1536
# ============================================================
print()
print('=== Test 4: Fix positions buffer to match N=1536 ===')
from borzoi_pytorch.pytorch_borzoi_transformer import get_positional_embed
x_196k = torch.randn(1, 4, 196608, device='cuda')
with torch.no_grad():
    x = model.conv_dna(x_196k)
    x_unet0 = model.res_tower(x)
    x_unet1 = model.unet1(x_unet0)
    x = model._max_pool(x_unet1)
    x = x.permute(0, 2, 1)
    N = x.shape[1]
    print(f'Sequence length N={N}')

    # Replace positions buffer in all transformer blocks
    new_positions = get_positional_embed(N, 32, x.device)
    print(f'New positions shape: {new_positions.shape} (was 8191)')
    for i, blk in enumerate(model.transformer):
        attn = blk[0].fn[1]
        attn.positions = new_positions

    # Now run transformer
    for i, blk in enumerate(model.transformer):
        x = blk(x)
        is_nan = torch.isnan(x).any().item()
        print(f'Block {i} (fixed pos): nan={is_nan} range=[{x.float().min():.2f}, {x.float().max():.2f}]')
        if is_nan:
            print(f'  Still NaN at block {i} even with fixed positions!')
            break

print('DONE')
"
