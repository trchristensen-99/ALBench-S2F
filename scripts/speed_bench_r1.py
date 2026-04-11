#!/usr/bin/env python
"""Round 1 LegNet speed benchmark — targeted optimization candidates.

Tests the following configurations on the H100 with N=296K training sequences:
  baseline        : AMP + compile(max-autotune) + CPU DataLoader (pin_memory=True)
  gpu_pinned      : AMP + compile(max-autotune) + all data on GPU (no per-batch transfer)
  no_compile      : AMP + no compile + GPU pinned (check compile overhead vs benefit)
  reduce_overhead : AMP + compile(reduce-overhead) + GPU pinned
  channels_last   : AMP + compile + GPU pinned + channels-last memory format
  bf16            : bfloat16 AMP (instead of float16) + compile + GPU pinned
  larger_bs       : AMP + compile + GPU pinned + bs=2048
  onecycle_lr     : Measure convergence speed (val_r @ epoch N) with OneCycleLR vs flat LR

Outputs wall time/epoch and val Pearson r for each config.

Usage (called from SLURM):
    uv run --no-sync python scripts/speed_bench_r1.py \
        --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \
        --n-train 296000 --epochs 12 \
        --output-dir outputs/legnet_speed_r1
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from torch.amp import GradScaler, autocast  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from models.legnet import LegNet, one_hot_encode_batch  # noqa: E402  # isort: skip


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pool_data(pool_path: str, n_train: int, seed: int = 42):
    """Load pool.npz → (train_seqs, train_labels, val_seqs, val_labels).

    Falls back to K562Dataset if pool.npz not found.
    """
    p = Path(pool_path)
    if p.exists():
        print(f"Loading pool from {p}")
        data = np.load(p, allow_pickle=True)
        sequences = data["sequences"].tolist()
        labels = data["labels"].astype(np.float32)
    else:
        print(f"Pool not found at {p}, falling back to K562Dataset")
        from data.k562 import K562Dataset

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


# ---------------------------------------------------------------------------
# GPU batch iterator (all data on GPU, zero CPU-GPU transfer per batch)
# ---------------------------------------------------------------------------


class GPUBatchIterator:
    def __init__(self, x, y, batch_size, shuffle=True, drop_last=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = x.shape[0]

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.x.device)
            x = self.x[idx]
            y = self.y[idx]
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
class BenchCfg:
    name: str
    batch_size: int = 512
    compile_mode: Optional[str] = "max-autotune"
    amp_dtype: str = "float16"  # "float16", "bfloat16", or "none"
    gpu_pinned: bool = True
    channels_last: bool = False
    lr: float = 0.005
    use_onecycle: bool = True  # True = OneCycleLR (as in production), False = flat LR
    description: str = ""


CONFIGS = [
    BenchCfg(
        name="baseline_cpu_loader",
        gpu_pinned=False,
        compile_mode="max-autotune",
        amp_dtype="float16",
        description="Current production: AMP fp16 + compile(max-autotune) + CPU DataLoader",
    ),
    BenchCfg(
        name="gpu_pinned",
        gpu_pinned=True,
        compile_mode="max-autotune",
        amp_dtype="float16",
        description="AMP fp16 + compile(max-autotune) + all data on GPU",
    ),
    BenchCfg(
        name="no_compile_gpu_pinned",
        gpu_pinned=True,
        compile_mode=None,
        amp_dtype="float16",
        description="AMP fp16 + NO compile + GPU pinned",
    ),
    BenchCfg(
        name="compile_reduce_overhead",
        gpu_pinned=True,
        compile_mode="reduce-overhead",
        amp_dtype="float16",
        description="AMP fp16 + compile(reduce-overhead) + GPU pinned",
    ),
    BenchCfg(
        name="bf16_compile",
        gpu_pinned=True,
        compile_mode="max-autotune",
        amp_dtype="bfloat16",
        description="bfloat16 AMP + compile(max-autotune) + GPU pinned (H100 native bf16)",
    ),
    BenchCfg(
        name="channels_last_compile",
        gpu_pinned=True,
        compile_mode="max-autotune",
        amp_dtype="float16",
        channels_last=True,
        description="AMP fp16 + compile + GPU pinned + channels-last memory layout",
    ),
    BenchCfg(
        name="bs2048_gpu_pinned",
        batch_size=2048,
        gpu_pinned=True,
        compile_mode="max-autotune",
        amp_dtype="float16",
        description="bs=2048 + AMP fp16 + compile + GPU pinned (H100 has plenty of VRAM)",
    ),
    BenchCfg(
        name="bf16_reduce_overhead",
        gpu_pinned=True,
        compile_mode="reduce-overhead",
        amp_dtype="bfloat16",
        description="bfloat16 AMP + compile(reduce-overhead) + GPU pinned",
    ),
]


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------


def make_model(device, channels_last=False):
    torch.manual_seed(42)
    m = LegNet(in_channels=4, task_mode="k562").to(device)
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def train_epoch(model, data_iter, optimizer, scaler, scheduler, cfg: BenchCfg, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
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
            p = model(xb).cpu()
            all_preds.append(p)
    preds = torch.cat(all_preds).numpy().reshape(-1)
    tgts = y_val.cpu().numpy().reshape(-1)
    r, _ = pearsonr(preds, tgts)
    return float(r)


# ---------------------------------------------------------------------------
# Run one benchmark config
# ---------------------------------------------------------------------------


def run_config(
    cfg: BenchCfg, x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, device, epochs, verbose=True
):
    print(f"\n{'=' * 72}")
    print(f"[{cfg.name}] {cfg.description}")
    print(f"  bs={cfg.batch_size}, compile={cfg.compile_mode}, amp={cfg.amp_dtype}")
    print(f"  gpu_pinned={cfg.gpu_pinned}, channels_last={cfg.channels_last}")

    model = make_model(device, channels_last=cfg.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Move data to GPU if pinned
    if cfg.gpu_pinned:
        x_tr = x_train_cpu.to(device)
        y_tr = y_train_cpu.to(device)
        x_vl = x_val_cpu.to(device)
        y_vl = y_val_cpu.to(device)
        torch.cuda.synchronize()
    else:
        x_tr, y_tr = x_train_cpu, y_train_cpu
        x_vl, y_vl = x_val_cpu, y_val_cpu

    # Compile
    if cfg.compile_mode is not None:
        t0 = time.perf_counter()
        model = torch.compile(model, mode=cfg.compile_mode)
        print(f"  compile() call: {time.perf_counter() - t0:.1f}s (warmup on 1st epoch)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)

    # Scheduler: OneCycleLR matching production code
    n_batches_per_epoch = x_tr.shape[0] // cfg.batch_size
    if cfg.use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            total_steps=n_batches_per_epoch * epochs,
            pct_start=0.3,
        )
    else:
        scheduler = None

    scaler = GradScaler("cuda", enabled=cfg.amp_dtype == "float16")

    epoch_times = []
    val_rs = []

    for ep in range(epochs):
        # Build data iterator
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

    # Skip first epoch (compile warmup)
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
    parser.add_argument(
        "--configs", nargs="*", default=None, help="Names of configs to run (default: all)"
    )
    parser.add_argument("--output-dir", default="outputs/legnet_speed_r1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")

    # Load data
    print(f"\nLoading data (n_train={args.n_train})...")
    train_seqs, train_labels, val_seqs, val_labels = load_pool_data(args.data_path, args.n_train)

    print("Encoding sequences (one-hot)...")
    t0 = time.perf_counter()
    x_train = encode_sequences(train_seqs, seq_len=200)
    y_train = torch.from_numpy(train_labels)
    x_val = encode_sequences(val_seqs, seq_len=200)
    y_val = torch.from_numpy(val_labels)
    print(
        f"  Encoding: {time.perf_counter() - t0:.1f}s | x_train={x_train.shape} x_val={x_val.shape}"
    )

    selected = CONFIGS
    if args.configs:
        selected = [c for c in CONFIGS if c.name in args.configs]

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

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY (Round 1)")
    print(f"{'=' * 80}")
    print(f"{'Config':<30s}  {'s/epoch (ep2+)':>14s}  {'1st epoch':>9s}  {'val_r':>7s}")
    print("-" * 65)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<30s}  {'ERROR':>14s}")
        else:
            first = f"{r['first_epoch_time_s']:.1f}s" if r.get("first_epoch_time_s") else "N/A"
            print(
                f"{name:<30s}  {r['mean_epoch_time_s']:>13.2f}s  {first:>9s}  {r['final_val_r']:>7.4f}"
            )

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results_r1.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
