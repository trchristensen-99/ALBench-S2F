"""Edge-case checks for --cl replay_real BEFORE committing a night of GPU time."""
import numpy as np, torch, sys
sys.path.insert(0, "scripts")
from fm_scaling_driver import _load_borzoi, borzoi_tracks, load_anchor, encode_all

dev = "cuda" if torch.cuda.is_available() else "cpu"
b = _load_borzoi().to(dev).eval()

# 1. does the ORIGINAL track head emit the same (tracks, bins) shape as our real targets?
x = torch.zeros(2, 4, 512).to(dev)
with torch.no_grad():
    out = borzoi_tracks(b, x)
print("borzoi_tracks(512bp) ->", tuple(out.shape))

z = np.load("outputs/replay_anchors/borzoi_real_n2048_w512.npz", allow_pickle=True)
tgt = z["track_targets"]
print("real targets        ->", (2, tgt.shape[1], tgt.shape[2]))
match = out.shape[1:] == tgt.shape[1:]
print("SHAPES_MATCH" if match else f"SHAPE_MISMATCH {tuple(out.shape[1:])} vs {tgt.shape[1:]}")

# 2. load_anchor with real targets + over-request (asks 5000, only 1024 exist)
s, t = load_anchor("outputs/replay_anchors/borzoi_real_n2048_w512.npz", 5000, 42, want_targets=True)
print(f"over-request handled: got n={len(s)} targets={t.shape}")

# 3. a missing-targets cache must fail LOUDLY, not silently train on nothing
try:
    load_anchor("outputs/chr_split_cache/chr_train_ref_only.npz", 8, 42, want_targets=True)
    print("PROBLEM: no error on cache without track_targets")
except ValueError as e:
    print("missing-targets raises correctly:", str(e)[:60])

# 4. one real joint step: MPRA loss + real-replay loss, verify finite grads
X = encode_all([str(x) for x in z["sequences"][:4]], 512, True)
gx = X.to(dev).float()
gt = torch.tensor(t[:4]).to(dev)
cur = borzoi_tracks(b, gx)
loss = torch.nn.functional.mse_loss(cur, gt)
loss.backward()
gnorm = sum(p.grad.norm().item() for p in b.parameters() if p.grad is not None)
print(f"replay loss={loss.item():.4f} grad_norm_finite={np.isfinite(gnorm)}")
print("EDGE_TESTS_DONE")
