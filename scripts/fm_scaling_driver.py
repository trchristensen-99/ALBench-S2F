"""Foundation-model scaling driver (the FM analog of the from-scratch CNN scaling laws).

For a (model x reservoir dataset x D x seed) cell: load a pretrained FM (Flashzoi/Borzoi/...),
attach a regression head predicting the oracle MPRA activity, fine-tune on D oracle-labeled
reservoir sequences, evaluate on the held-out battery (+ eQTL). Records the scaling point (perf vs D).

Short-sequence (200 bp) handling: use the FM's CONV TRUNK embedding (Borzoi: conv_dna -> res_tower),
which is length-agnostic, then global-pool + Linear head — the same trick the Enformer MPRA head uses
(it drops the long-range transformer/crop that require ~500 kb inputs). The encoder stays intact so a
continual-learning wrapper (clgenomics) can attach extra heads later.

Two-stage fine-tune: (1) frozen trunk + head warmup, (2) full fine-tune. CL variant (--cl replay) via
clgenomics.ReplayContinualLearner is added AFTER the non-CL curves (per PI sequencing).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

_BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
_CODE_LUT = np.full(256, -1, dtype=np.int64)  # ASCII code -> base index (-1 for N/other)
for _b, _i in _BASE2IDX.items():
    _CODE_LUT[ord(_b)] = _i
FLASHZOI_REPO = os.environ.get("FLASHZOI_REPO", "johahi/flashzoi-replicate-0")
BORZOI_REPO = os.environ.get("BORZOI_REPO", "johahi/borzoi-replicate-0")


def one_hot(seqs, length=None, center=False):
    """list[str] -> (N, 4, L) float tensor (channels-first for conv stems).
    center=True places each sequence in the middle of the L-window (zero-padded flanks) — used by the
    full-encoder head so the MPRA element sits in the center window that the pooling head reads.
    Vectorized per sequence (no per-base Python loop) to keep CPU off the critical path."""
    L = length or max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), 4, L)
    for i, s in enumerate(seqs):
        s = s[:L]
        off = max(0, (L - len(s)) // 2) if center else 0
        codes = _CODE_LUT[np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)]
        pos = np.nonzero(codes >= 0)[0]  # drop N / non-ACGT
        x[i, codes[pos], pos + off] = 1.0
    return x


def borzoi_encode(b, x):
    """Replicate Borzoi.get_embs_after_crop WITHOUT the final TargetLengthCrop -> (N, config.dim, L/16)."""
    x = b.conv_dna(x)
    x_unet0 = b.res_tower(x)
    x_unet1 = b.unet1(x_unet0)
    x = b._max_pool(x_unet1)
    x_unet1 = b.horizontal_conv1(x_unet1)  # project skip to config.dim (1536)
    x_unet0 = b.horizontal_conv0(x_unet0)  # project skip 1280 -> config.dim (1536)
    x = b.transformer(x.permute(0, 2, 1))
    x = x.permute(0, 2, 1)
    x = b.upsampling_unet1(x)
    x = x + x_unet1
    x = b.separable1(x)
    x = b.upsampling_unet0(x)
    x = x + x_unet0
    x = b.separable0(x)
    return x


def borzoi_tracks(b, x):
    """Original Borzoi track-prediction path on a short (skip-crop) window -> (N, n_tracks, bins).
    Used for continual-learning distillation/replay: keep the fine-tuned encoder's genomic-track
    outputs close to the frozen original model's on genomic windows."""
    x = borzoi_encode(b, x)
    x = b.final_joined_convs(x)
    return b.final_softplus(b.human_head(x))


class BorzoiTrunkRegressor(nn.Module):
    """Borzoi conv trunk (conv_dna -> res_tower) -> global-avg-pool -> Linear head."""

    def __init__(self, borzoi, head_dim=1):
        super().__init__()
        self.borzoi = borzoi
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 512)
            emb = borzoi.res_tower(borzoi.conv_dna(dummy))  # (1, C, L')
        self.embed_dim = emb.shape[1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.embed_dim, head_dim)
        )

    def trunk(self, x):
        return self.borzoi.res_tower(self.borzoi.conv_dna(x))

    def forward(self, x):
        return self.head(self.trunk(x)).squeeze(-1)

    def set_encoder_trainable(self, flag):
        for p in self.borzoi.parameters():
            p.requires_grad = flag


class BorzoiFullEncoderRegressor(nn.Module):
    """FULL Borzoi encoder (conv_dna -> res_tower -> unet1 -> transformer -> upsampling U-Net ->
    separable), MINUS the fixed 16352-bin crop, -> center-window pool -> MLP head.

    This mirrors the AlphaGenome MPRA encoder-only fine-tune (full sequence_encoder + a center-window
    pooling head), and — unlike the conv-trunk head — keeps the ENTIRE model (incl. transformer) and
    its original human_head intact, so a continual-learning wrapper can replay Borzoi's original
    genomic-track task through the original head while this MPRA head is added. Bypassing the crop lets
    us run the full transformer on a moderate padded input (~512 bp) instead of a ~524 kb window.
    """

    def __init__(
        self, borzoi, head_dim=1, center_bins=None, pooling="mean", hidden=256, dropout=0.1
    ):
        super().__init__()
        self.borzoi = borzoi
        self.pooling = pooling
        self.center_bins = center_bins  # None -> pool all bins
        with torch.no_grad():
            emb = self._encode(torch.zeros(1, 4, 512))  # (1, C, Lb)
        self.embed_dim = emb.shape[1]
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, head_dim),
        )

    def _encode(self, x):
        return borzoi_encode(self.borzoi, x)

    def forward(self, x):
        emb = self._encode(x)  # (N, C, Lb)
        if self.center_bins:
            lb = emb.shape[-1]
            c, h = lb // 2, self.center_bins // 2
            emb = emb[..., max(0, c - h) : c + h + (self.center_bins % 2)]
        if self.pooling == "mean":
            pooled = emb.mean(-1)
        elif self.pooling == "sum":
            pooled = emb.sum(-1)
        else:
            pooled = emb.max(-1).values
        return self.head(pooled).squeeze(-1)

    def set_encoder_trainable(self, flag):
        for p in self.borzoi.parameters():
            p.requires_grad = flag


def _load_borzoi():
    from borzoi_pytorch import Borzoi

    # Load the cached borzoi repo with flashed=False -> standard attention, no flash_attn dependency
    # (flashzoi weights are identical; FlashAttention only accelerates the transformer). Install
    # flash_attn later purely for speed if desired.
    try:
        return Borzoi.from_pretrained(BORZOI_REPO, flashed=False)
    except TypeError:
        b = Borzoi.from_pretrained(BORZOI_REPO)
        b.flashed = False
        return b


def load_fm(
    model_name,
    head="full_encoder",
    head_dim=1,
    center_bins=None,
    pooling="mean",
    hidden=256,
    dropout=0.1,
):
    if model_name in ("flashzoi", "borzoi"):
        borzoi = _load_borzoi()
        if head == "trunk":
            return BorzoiTrunkRegressor(borzoi, head_dim)
        return BorzoiFullEncoderRegressor(
            borzoi,
            head_dim,
            center_bins=center_bins,
            pooling=pooling,
            hidden=hidden,
            dropout=dropout,
        )
    raise NotImplementedError(f"wire {model_name} (NTv3 / AG-fold0) next")


def load_reservoir(cache_path, D, seed):
    z = np.load(cache_path, allow_pickle=True)
    key = "sequences"
    seqs = z[key]
    labels = z["oracle_labels"].astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=min(D, len(seqs)), replace=False)
    return [str(seqs[i]) for i in idx], labels[idx]


def load_battery(battery_dir, limit=None):
    out = {}
    for f in sorted(os.listdir(battery_dir)):
        if not f.endswith("_oracle.npz"):
            continue
        z = np.load(os.path.join(battery_dir, f), allow_pickle=True)
        lab = next((k for k in ("oracle_mean", "oracle_labels") if k in z.files), None)
        if "sequences" in z.files and lab:
            s = [str(x) for x in z["sequences"]]
            y = z[lab].astype(np.float32)
            if limit:
                s, y = s[:limit], y[:limit]
            out[f.replace("_oracle.npz", "")] = (s, y)
    return out


def encode_all(seqs, L, center):
    """One-hot the whole split ONCE as uint8 (N,4,L) — the per-batch re-encoding was the real
    bottleneck (CPU-bound, repeated every epoch). uint8 keeps it small: D=300k @512bp = 0.6 GB."""
    return one_hot(seqs, L, center=center).to(torch.uint8)


def _batches(X, labels, bs, device, shuffle=False):
    """X: precomputed uint8 (N,4,L) one-hot. Batches just index + cast on device."""
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        xb = X[b].to(device, non_blocking=True).float()
        yb = torch.tensor(labels[b], device=device)
        yield xb, yb


def _cycle_anchor(X, targets, bs, device):
    """Infinite stream of genomic anchor batches from precomputed one-hot X (+ optional targets)."""
    while True:
        order = np.random.permutation(len(X))
        for i in range(0, len(order), bs):
            b = order[i : i + bs]
            gx = X[b].to(device, non_blocking=True).float()
            gt = None if targets is None else torch.tensor(targets[b], device=device)
            yield gx, gt


@torch.no_grad()
def _val_pearson(model, Xv, yv, device, amp):
    """Held-out val correlation — the ONLY legitimate signal for HP selection (the battery is test)."""
    model.eval()
    preds = []
    for xb, _ in _batches(Xv, np.zeros(len(Xv), np.float32), 128, device):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            out = model(xb)
        preds.append(out.float().cpu().numpy())
    p = np.concatenate(preds)
    m = np.isfinite(p) & np.isfinite(yv)
    model.train()
    return float(pearsonr(p[m], yv[m])[0]) if m.sum() > 3 else float("nan")


def train_fm(model, X, labels, args, device, ref_borzoi=None, anchor=None, val=None):
    """X = precomputed one-hot for the training split. anchor = (anchor_X, targets_or_None):
    distill = match frozen ref model's track outputs; replay_real = match provided real track targets.
    Anchoring runs jointly (multitask) with the MPRA loss, only in stage-2 (once the encoder unfreezes
    — nothing to preserve while it is frozen)."""
    labels = np.asarray(labels)
    amp = args.amp and device == "cuda"  # bf16 autocast: ~2x throughput + ~half memory on H100
    wd = getattr(args, "weight_decay", 1e-4)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=wd
    )
    lossf = nn.MSELoss()
    stage1 = max(1, int(round(args.epochs * getattr(args, "stage1_frac", 0.25))))
    best_val, best_ep = float("-inf"), -1
    anchor_on = args.cl in ("distill", "replay_real") and anchor is not None
    model.set_encoder_trainable(False)  # stage 1: head only
    for ep in range(args.epochs):
        if ep == stage1:  # stage 2: full fine-tune (encoder at a reduced LR)
            model.set_encoder_trainable(True)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * getattr(args, "encoder_lr_mult", 0.1),
                weight_decay=wd,
            )
        model.train()
        astream = (
            _cycle_anchor(anchor[0], anchor[1], args.batch_size, device)
            if (anchor_on and ep >= stage1)
            else None
        )
        tot = 0.0
        dtot = 0.0
        for xb, yb in _batches(X, labels, args.batch_size, device, shuffle=True):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = lossf(model(xb), yb)
                if astream is not None:
                    gx, gt = next(astream)
                    cur = borzoi_tracks(model.borzoi, gx)
                    if gt is None:  # distill: target = frozen original model's outputs
                        with torch.no_grad():
                            gt = borzoi_tracks(ref_borzoi, gx)
                    dloss = lossf(cur, gt)
                    loss = loss + args.replay_lambda * dloss
                    dtot += dloss.item()
            loss.backward()
            opt.step()
            tot += loss.item()
        vmsg = ""
        if val is not None and (ep % 5 == 0 or ep == args.epochs - 1):
            v = _val_pearson(model, val[0], val[1], device, amp)
            if v > best_val:
                best_val, best_ep = v, ep
            vmsg = f" val_r={v:.4f}"
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(
                f"  ep{ep} stage{1 if ep < stage1 else 2} loss={tot:.3f} distill={dtot:.3f}{vmsg}",
                flush=True,
            )
    return {"best_val_pearson": best_val if best_ep >= 0 else None, "best_val_epoch": best_ep}


@torch.no_grad()
def genomic_preservation(borzoi_ft, ref_borzoi, anchor_X, device, n=1024):
    """Pearson between the fine-tuned encoder's and the frozen original model's track outputs on
    held-out genomic windows. High -> original genomic behavior preserved (low catastrophic forgetting).
    Reported for every arm (incl. cl=none) so backward-transfer arms can be compared against naive FT."""
    ref_borzoi.eval()
    cur, ref = [], []
    for i in range(0, min(n, len(anchor_X)), 32):
        gx = anchor_X[i : i + 32].to(device).float()
        cur.append(borzoi_tracks(borzoi_ft, gx).float().cpu().numpy().ravel())
        ref.append(borzoi_tracks(ref_borzoi, gx).float().cpu().numpy().ravel())
    c, r = np.concatenate(cur), np.concatenate(ref)
    m = np.isfinite(c) & np.isfinite(r)
    return float(pearsonr(c[m], r[m])[0])


def load_anchor(path, n, seed, want_targets=False):
    """Genomic anchor windows for CL. want_targets -> also return real track targets (replay_real)."""
    z = np.load(path, allow_pickle=True)
    seqs = z["sequences"]
    rng = np.random.default_rng(seed + 9999)
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    s = [str(seqs[i]) for i in idx]
    t = None
    if want_targets:
        tkey = next((k for k in ("track_targets", "targets", "tracks") if k in z.files), None)
        if tkey is None:
            raise ValueError(
                f"replay_real needs real track targets in {path} (have {list(z.files)})"
            )
        t = z[tkey][idx].astype(np.float32)
    return s, t


@torch.no_grad()
def evaluate(model, battery, device, L, center, amp=False):
    model.eval()
    metrics = {}
    for name, (seqs, y) in battery.items():
        X = encode_all(seqs, L, center)
        preds = []
        for xb, _ in _batches(X, np.zeros(len(seqs), np.float32), 64, device):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
                out = model(xb)
            preds.append(out.float().cpu().numpy())
        p = np.concatenate(preds)
        m = np.isfinite(p) & np.isfinite(y)
        if m.sum() > 3:
            metrics[name] = float(pearsonr(p[m], y[m])[0])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="flashzoi", choices=["flashzoi", "borzoi", "ntv3", "alphagenome_fold0"]
    )
    ap.add_argument("--reservoir_cache")
    ap.add_argument(
        "--genomic_train",
        help="use chr_train_ref_only.npz-style npz (sequences+oracle_labels) instead of a reservoir cache",
    )
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--battery_dir", default="data/k562/test_sets_ag_s2_chrsplit")
    ap.add_argument(
        "--battery_limit", type=int, default=None, help="cap test-set size (smoke tests)"
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--cl",
        choices=["none", "distill", "replay_real"],
        default="none",
        help="none = MPRA-only full FT (may forget); distill = joint MPRA + KD to frozen Borzoi's track "
        "outputs on genomic windows (functional replay, no original labels needed); replay_real = joint "
        "MPRA + real Borzoi track targets (needs --anchor_cache with targets)",
    )
    ap.add_argument(
        "--replay_lambda", type=float, default=1.0, help="weight on the anchor/distill loss"
    )
    ap.add_argument(
        "--anchor_cache",
        default="outputs/chr_split_cache/chr_train_ref_only.npz",
        help="genomic windows for the CL anchor (and preservation metric)",
    )
    ap.add_argument("--anchor_n", type=int, default=4096)
    ap.add_argument(
        "--measure_preservation",
        action="store_true",
        help="report genomic-track preservation vs frozen original (loads a frozen ref; implied by --cl distill)",
    )
    ap.add_argument(
        "--head",
        choices=["full_encoder", "trunk"],
        default="full_encoder",
        help="full_encoder = AG-style full Borzoi encoder + center-window pool (CL-ready, default); "
        "trunk = cheap conv-trunk-only (separate comparison curve, not CL-ready)",
    )
    ap.add_argument(
        "--input_len",
        type=int,
        default=512,
        help="padded input length for full_encoder (must be divisible by the encoder downsampling; "
        "the ~200 bp element is centered)",
    )
    ap.add_argument(
        "--center_bins",
        type=int,
        default=None,
        help="pool only the center N encoder bins (default: all)",
    )
    ap.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean")
    ap.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="bf16 autocast on CUDA (~2x throughput, ~half memory); on by default",
    )
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    # --- fine-tuning HPs. Tuned ONCE on a reservoir-balanced mixture and then FROZEN across every
    # cell, so the reservoir/acquisition comparison is never confounded by per-cell tuning. ---
    ap.add_argument(
        "--encoder_lr_mult", type=float, default=0.1, help="stage-2 encoder LR = lr * this"
    )
    ap.add_argument(
        "--stage1_frac", type=float, default=0.25, help="fraction of epochs with frozen encoder"
    )
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--head_dropout", type=float, default=0.1)
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="held-out fraction of TRAIN for HP selection (never select on the battery = test)",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    src = args.reservoir_cache or args.genomic_train
    seqs, labels = load_reservoir(src, args.D, args.seed)
    # full_encoder centers the element in a fixed padded window; trunk uses the native element length
    center = args.head == "full_encoder"
    L = args.input_len if center else max(len(s) for s in seqs)
    battery = load_battery(args.battery_dir, args.battery_limit)
    print(
        f"[fm] model={args.model} head={args.head} D={len(seqs)} L={L} center={center} "
        f"sets={list(battery)} device={device}",
        flush=True,
    )

    model = load_fm(
        args.model,
        head=args.head,
        center_bins=args.center_bins,
        pooling=args.pooling,
        hidden=args.head_hidden,
        dropout=args.head_dropout,
    ).to(device)
    print(f"[fm] {args.model} embed_dim={model.embed_dim}", flush=True)

    # Continual-learning anchor: frozen reference model + genomic windows (needed for distill/replay
    # and for the preservation metric, which we report for every arm to quantify forgetting).
    ref_borzoi, anchor = None, None
    need_ref = args.cl in ("distill", "replay_real") or args.measure_preservation
    if need_ref:
        ref_borzoi = _load_borzoi().to(device).eval()
        for p in ref_borzoi.parameters():
            p.requires_grad = False
        a_seqs, a_tgt = load_anchor(
            args.anchor_cache, args.anchor_n, args.seed, want_targets=(args.cl == "replay_real")
        )
        anchor = (encode_all(a_seqs, L, center), a_tgt)
        print(f"[fm] cl={args.cl} lambda={args.replay_lambda} anchor_n={len(a_seqs)}", flush=True)

    X = encode_all(seqs, L, center)  # encode the training split once (not per batch per epoch)
    print(
        f"[fm] encoded train one-hot {tuple(X.shape)} ({X.numel() / 1e9:.2f} GB uint8)", flush=True
    )
    # Hold out a val split from TRAIN for HP selection — the battery is TEST, and selecting on it
    # would leak. Seeded so every arm of a comparison sees the same partition.
    val = None
    if args.val_frac and args.val_frac > 0:
        perm = np.random.default_rng(args.seed + 777).permutation(len(X))
        nv = max(1, int(len(X) * args.val_frac))
        vi, ti = perm[:nv], perm[nv:]
        lab = np.asarray(labels)
        val = (X[vi], lab[vi])
        X, labels = X[ti], lab[ti]
        print(f"[fm] train={len(X)} val={len(vi)} (val_frac={args.val_frac})", flush=True)

    t0 = time.time()
    fit = train_fm(model, X, labels, args, device, ref_borzoi=ref_borzoi, anchor=anchor, val=val)
    metrics = evaluate(model, battery, device, L, center, amp=args.amp)
    preservation = None
    if need_ref:
        preservation = genomic_preservation(model.borzoi, ref_borzoi, anchor[0], device)
        print(f"[fm] genomic_preservation={preservation:.4f}", flush=True)
    out = {
        "model": args.model,
        "head": args.head,
        "input_len": L,
        "pooling": args.pooling,
        "center_bins": args.center_bins,
        "D": len(seqs),
        "cl": args.cl,
        "replay_lambda": args.replay_lambda,
        "hp": {
            "lr": args.lr,
            "encoder_lr_mult": args.encoder_lr_mult,
            "stage1_frac": args.stage1_frac,
            "weight_decay": args.weight_decay,
            "head_hidden": args.head_hidden,
            "head_dropout": args.head_dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "input_len": args.input_len,
            "pooling": args.pooling,
            "center_bins": args.center_bins,
        },
        "val_pearson": (fit or {}).get("best_val_pearson"),
        "val_best_epoch": (fit or {}).get("best_val_epoch"),
        "genomic_preservation": preservation,
        "train_sec": round(time.time() - t0, 1),
        "metrics": metrics,
    }
    json.dump(out, open(os.path.join(args.out_dir, "fm_scaling_point.json"), "w"), indent=2)
    print(
        f"[fm] DONE {args.model} D={len(seqs)} genomic={metrics.get('genomic')} -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
