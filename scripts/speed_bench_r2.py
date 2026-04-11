#!/usr/bin/env python
"""Round 2 LegNet speed benchmark — targeted follow-up optimizations.

Focuses on the most promising directions from Round 1:
  1. bf16 + channels_last combined
  2. Fused AdamW (torch.optim.AdamW with fused=True on H100)
  3. Aggressive DataLoader: num_workers tuning + prefetch_factor
  4. Scripted model (torch.jit.script) vs compile
  5. torch.compile with fullgraph=True
  6. gradient_as_bucket_view + set_to_none optimization
  7. Pre-computed RC aug baked into dataset (avoids per-batch flip)
  8. Best-of-R1 combined with all micro-opts

Usage:
    uv run --no-sync python scripts/speed_bench_r2.py \
        --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \
        --n-train 296000 --epochs 12 \
        --winner <name_from_r1> \
        --output-dir outputs/legnet_speed_r2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from torch.amp import GradScaler, autocast  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from models.legnet import LegNet, one_hot_encode_batch  # noqa: E402  # isort: skip


# ---------------------------------------------------------------------------
# Data helpers (shared with r1)
# ---------------------------------------------------------------------------


def load_pool_data(pool_path: str, n_train: int, seed: int = 42):
    p = Path(pool_path)
    if p.exists():
        print(f"Loading pool from {p}")
        data = np.load(p, allow_pickle=True)
        sequences = data["sequences"].tolist()
        labels = data["labels"].astype(np.float32)
    else:
        print(f"Pool not found at {p}, falling back to K562Dataset")
        from data.k562 import K562Dataset  # isort: skip

        ds = K562Dataset(
            data_path=str(REPO / "data" / "k562"),
            split="train",
            label_column="K562_log2FC",
            use_hashfrag=True,
        )
        sequences = list(ds.sequences)
        labels = ds.labels.astype(np.float32)

    rng = np.random.RandomState(seed)
    n = len(sequences)
    idx = rng.permutation(n)
    n_val = min(30_000, max(5_000, int(n * 0.1)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val : n_val + n_train]
    val_seqs = [sequences[i] for i in val_idx]
    val_labels = labels[val_idx]
    train_seqs = [sequences[i] for i in train_idx]
    train_labels = labels[train_idx]
    print(f"  Train: {len(train_seqs)}, Val: {len(val_seqs)}")
    return train_seqs, train_labels, val_seqs, val_labels


def encode_sequences(sequences: list[str], seq_len: int = 200) -> torch.Tensor:
    standardized = []
    for seq in sequences:
        seq = seq.upper()
        if len(seq) < seq_len:
            pad = seq_len - len(seq)
            seq = "N" * (pad // 2) + seq + "N" * (pad - pad // 2)
        elif len(seq) > seq_len:
            start = (len(seq) - seq_len) // 2
            seq = seq[start : start + seq_len]
        standardized.append(seq)
    arr = one_hot_encode_batch(standardized, seq_len=seq_len)
    return torch.from_numpy(arr)


class GPUBatchIterator:
    def __init__(self, x, y, batch_size, shuffle=True, drop_last=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = x.shape[0]

    def __len__(self):
        return (
            self.n // self.batch_size
            if self.drop_last
            else (self.n + self.batch_size - 1) // self.batch_size
        )

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.x.device)
            x, y = self.x[idx], self.y[idx]
        else:
            x, y = self.x, self.y
        for i in range(0, self.n, self.batch_size):
            if self.drop_last and i + self.batch_size > self.n:
                break
            yield x[i : i + self.batch_size], y[i : i + self.batch_size]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class R2Cfg:
    name: str
    batch_size: int = 512
    compile_mode: Optional[str] = "max-autotune"
    compile_fullgraph: bool = False
    amp_dtype: str = "bfloat16"  # winner is likely bf16 from R1
    gpu_pinned: bool = True
    channels_last: bool = False
    fused_adamw: bool = False
    lr: float = 0.005
    description: str = ""


# R1 winner assumed to be bf16+compile+gpu_pinned — used as baseline for R2 comparisons
R2_CONFIGS = [
    # Baseline: best from R1 (assumed bf16 + compile + gpu_pinned)
    R2Cfg(
        name="r1_winner_bf16",
        amp_dtype="bfloat16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        description="R1 winner: bf16 + compile(max-autotune) + GPU pinned",
    ),
    # Fused AdamW — uses CUDA kernel fusion for optimizer step (H100 benefits)
    R2Cfg(
        name="fused_adamw",
        amp_dtype="bfloat16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        fused_adamw=True,
        description="bf16 + compile + GPU pinned + fused AdamW",
    ),
    # bf16 + channels_last combined
    R2Cfg(
        name="bf16_channels_last",
        amp_dtype="bfloat16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        channels_last=True,
        description="bf16 + channels_last + compile + GPU pinned",
    ),
    # compile with fullgraph=True (eliminate graph breaks)
    R2Cfg(
        name="compile_fullgraph",
        amp_dtype="bfloat16",
        compile_mode="max-autotune",
        compile_fullgraph=True,
        gpu_pinned=True,
        description="bf16 + compile(max-autotune, fullgraph=True) + GPU pinned",
    ),
    # fused_adamw + channels_last + bf16 (kitchen sink)
    R2Cfg(
        name="bf16_cl_fused",
        amp_dtype="bfloat16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        channels_last=True,
        fused_adamw=True,
        description="bf16 + channels_last + fused AdamW + compile + GPU pinned",
    ),
    # larger batch size with bf16 (H100 has 80GB VRAM)
    R2Cfg(
        name="bf16_bs2048",
        batch_size=2048,
        amp_dtype="bfloat16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        fused_adamw=True,
        description="bf16 + bs=2048 + fused AdamW + compile + GPU pinned",
    ),
    # fp16 + fused adamw (compare with bf16)
    R2Cfg(
        name="fp16_fused",
        amp_dtype="float16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        fused_adamw=True,
        description="fp16 + fused AdamW + compile + GPU pinned",
    ),
]


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def make_model(device, channels_last=False):
    torch.manual_seed(42)
    m = LegNet(in_channels=4, task_mode="k562").to(device)
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def train_epoch(model, data_iter, optimizer, scaler, scheduler, cfg: R2Cfg, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = cfg.amp_dtype != "none"

    for x, y in data_iter:
        if not cfg.gpu_pinned:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.to(memory_format=torch.channels_last)

        if use_amp:
            with autocast("cuda", dtype=amp_dtype):
                pred = model(x)
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def eval_model(model, x_val, y_val, batch_size, gpu_pinned, channels_last, device):
    model.eval()
    all_preds = []
    n = x_val.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x_val[i : i + batch_size]
            if not gpu_pinned:
                xb = xb.to(device)
            if channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            all_preds.append(model(xb).cpu())
    preds = torch.cat(all_preds).numpy().reshape(-1)
    tgts = y_val.cpu().numpy().reshape(-1)
    r, _ = pearsonr(preds, tgts)
    return float(r)


def run_config(cfg: R2Cfg, x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, device, epochs):
    print(f"\n{'=' * 72}")
    print(f"[{cfg.name}] {cfg.description}")
    print(f"  bs={cfg.batch_size}, compile={cfg.compile_mode}, fullgraph={cfg.compile_fullgraph}")
    print(
        f"  amp={cfg.amp_dtype}, fused_adamw={cfg.fused_adamw}, channels_last={cfg.channels_last}"
    )

    model = make_model(device, channels_last=cfg.channels_last)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    if cfg.gpu_pinned:
        x_tr = x_train_cpu.to(device)
        y_tr = y_train_cpu.to(device)
        x_vl = x_val_cpu.to(device)
        y_vl = y_val_cpu.to(device)
        torch.cuda.synchronize()
    else:
        x_tr, y_tr = x_train_cpu, y_train_cpu
        x_vl, y_vl = x_val_cpu, y_val_cpu

    if cfg.compile_mode is not None:
        t0 = time.perf_counter()
        model = torch.compile(model, mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph)
        print(f"  compile() call: {time.perf_counter() - t0:.1f}s")

    # Optimizer
    if cfg.fused_adamw:
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=0.01, fused=True
            )
            print("  Using fused AdamW")
        except TypeError:
            # older PyTorch without fused kwarg
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
            print("  fused AdamW not available, falling back to standard AdamW")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    optimizer.zero_grad(set_to_none=True)

    n_batches_per_epoch = x_tr.shape[0] // cfg.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=n_batches_per_epoch * epochs,
        pct_start=0.3,
    )

    # GradScaler only needed for fp16; bf16 doesn't need scaling
    use_scaler = cfg.amp_dtype == "float16"
    scaler = GradScaler("cuda", enabled=use_scaler)

    epoch_times = []
    val_rs = []

    for ep in range(epochs):
        if cfg.gpu_pinned:
            data_iter = GPUBatchIterator(x_tr, y_tr, cfg.batch_size, shuffle=True, drop_last=True)
        else:
            ds = TensorDataset(x_tr, y_tr)
            data_iter = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4,
                persistent_workers=True,
                drop_last=True,
            )

        t0 = time.perf_counter()
        train_loss = train_epoch(model, data_iter, optimizer, scaler, scheduler, cfg, device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        epoch_times.append(t1 - t0)

        val_r = eval_model(
            model, x_vl, y_vl, cfg.batch_size, cfg.gpu_pinned, cfg.channels_last, device
        )
        val_rs.append(val_r)
        print(f"  ep{ep + 1:02d}: {t1 - t0:.2f}s  loss={train_loss:.4f}  val_r={val_r:.4f}")

    mean_t = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else epoch_times[0]
    final_r = val_rs[-1]
    print(f"  => mean epoch (ep2+): {mean_t:.2f}s | final val_r: {final_r:.4f}")
    return {
        "mean_epoch_time_s": mean_t,
        "first_epoch_time_s": epoch_times[0],
        "epoch_times": epoch_times,
        "val_r_history": val_rs,
        "final_val_r": final_r,
        "config": asdict(cfg),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="outputs/labeled_pools/k562/ag_s2/genomic/pool.npz")
    parser.add_argument("--n-train", type=int, default=296000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument(
        "--winner", type=str, default=None, help="Best config name from R1 (for annotation only)"
    )
    parser.add_argument("--output-dir", default="outputs/legnet_speed_r2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")

    if args.winner:
        print(f"\nR1 winner (from arg): {args.winner}")

    print(f"\nLoading data (n_train={args.n_train})...")
    train_seqs, train_labels, val_seqs, val_labels = load_pool_data(args.data_path, args.n_train)

    print("Encoding sequences...")
    t0 = time.perf_counter()
    x_train = encode_sequences(train_seqs, seq_len=200)
    y_train = torch.from_numpy(train_labels)
    x_val = encode_sequences(val_seqs, seq_len=200)
    y_val = torch.from_numpy(val_labels)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    selected = R2_CONFIGS
    if args.configs:
        selected = [c for c in R2_CONFIGS if c.name in args.configs]

    results = {}
    for cfg in selected:
        try:
            r = run_config(cfg, x_train, y_train, x_val, y_val, device, args.epochs)
            results[cfg.name] = r
        except Exception as e:
            import traceback

            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[cfg.name] = {"error": str(e)}
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print(f"\n{'=' * 80}")
    print("SUMMARY (Round 2)")
    print(f"{'=' * 80}")
    print(f"{'Config':<30s}  {'s/epoch (ep2+)':>14s}  {'1st epoch':>9s}  {'val_r':>7s}")
    print("-" * 65)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<30s}  {'ERROR':>14s}")
        else:
            first = f"{r['first_epoch_time_s']:.1f}s"
            print(
                f"{name:<30s}  {r['mean_epoch_time_s']:>13.2f}s  {first:>9s}  {r['final_val_r']:>7.4f}"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results_r2.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
