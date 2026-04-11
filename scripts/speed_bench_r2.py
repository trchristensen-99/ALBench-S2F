#!/usr/bin/env python
"""Round 2 LegNet speed benchmark — targeted follow-up optimizations.

Round 1 findings (H100 NVL, PyTorch 2.10, N=296K, bs=512):
  - Baseline (fp16+compile+CPU loader):  42.06s/epoch, val_r=0.9454  <-- current production
  - GPU pinned (bs=512):                 41.84s/epoch  (~0.5% faster — minimal)
  - No compile:                          46.85s/epoch  (11% SLOWER — compile is essential)
  - reduce-overhead:                     44.40s/epoch  (worse than max-autotune)
  - bf16:                                42.94s/epoch  (SLOWER than fp16 — no benefit)
  - channels_last:                       FAILED (Conv1d is 3D, not 4D)
  - bs=2048:                             40.19s/epoch  (FASTEST at 4.4% speedup)
  - bf16_reduce_overhead:                43.50s, val_r=0.36 (BROKEN — bf16+reduce-overhead diverges)

Round 1 winner: bs=2048 + fp16 + compile(max-autotune) + GPU pinned = 40.19s/epoch
Key finding: bottleneck is compute, not data transfer (GPU pinned barely helps).

Round 2 focuses on:
  1. Fused AdamW (true optimizer fusion, should save ~1s/epoch)
  2. bs=1024 (midpoint — likely 41s, same quality as bs=512)
  3. bs=2048 + fused AdamW (potentially 38s?)
  4. torch.compile(dynamic=True) to skip graph recompile on first epoch
  5. compile + fullgraph=True (eliminate Python graph breaks)
  6. Gradient accumulation equivalent: bs=512 with accum=4 vs native bs=2048
  7. Combined winner: fp16 + bs=2048 + fused AdamW + gpu_pinned

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
    compile_dynamic: bool = False
    amp_dtype: str = "float16"  # R1 winner: fp16 is faster than bf16
    gpu_pinned: bool = True
    fused_adamw: bool = False
    grad_accum: int = 1  # gradient accumulation steps
    lr: float = 0.005
    description: str = ""


# R1 winner: fp16 + compile(max-autotune) + gpu_pinned + bs=2048 = 40.19s/epoch
R2_CONFIGS = [
    # R1 winner (confirmed baseline for R2)
    R2Cfg(
        name="r1_winner_bs2048",
        batch_size=2048,
        amp_dtype="float16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        description="R1 winner: fp16 + compile(max-autotune) + GPU pinned + bs=2048",
    ),
    # bs=1024 as midpoint (should be ~41s, same quality as bs=512)
    R2Cfg(
        name="bs1024_fp16",
        batch_size=1024,
        amp_dtype="float16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        description="fp16 + compile + GPU pinned + bs=1024",
    ),
    # Fused AdamW with bs=512 (pure optimizer speedup, baseline quality)
    R2Cfg(
        name="fused_adamw_bs512",
        batch_size=512,
        amp_dtype="float16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        fused_adamw=True,
        description="fp16 + compile + GPU pinned + fused AdamW + bs=512",
    ),
    # Fused AdamW with bs=2048 (best compute + optimizer fusion)
    R2Cfg(
        name="fused_adamw_bs2048",
        batch_size=2048,
        amp_dtype="float16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        fused_adamw=True,
        description="fp16 + compile + GPU pinned + fused AdamW + bs=2048",
    ),
    # torch.compile with dynamic=True (avoids per-epoch recompile if sequence length varies)
    R2Cfg(
        name="compile_dynamic_bs512",
        batch_size=512,
        amp_dtype="float16",
        compile_mode="max-autotune",
        compile_dynamic=True,
        gpu_pinned=True,
        description="fp16 + compile(max-autotune, dynamic=True) + GPU pinned + bs=512",
    ),
    # torch.compile with fullgraph=True (eliminate Python interpreter overhead)
    R2Cfg(
        name="compile_fullgraph_bs512",
        batch_size=512,
        amp_dtype="float16",
        compile_mode="max-autotune",
        compile_fullgraph=True,
        gpu_pinned=True,
        description="fp16 + compile(max-autotune, fullgraph=True) + GPU pinned + bs=512",
    ),
    # Gradient accumulation: bs=512 x4 (eff bs=2048) — same effective batch, same wall time?
    R2Cfg(
        name="grad_accum_4x",
        batch_size=512,
        amp_dtype="float16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        grad_accum=4,
        description="fp16 + compile + GPU pinned + bs=512 + grad_accum=4 (eff bs=2048)",
    ),
    # bs=4096 — push batch size even further
    R2Cfg(
        name="bs4096_fp16",
        batch_size=4096,
        amp_dtype="float16",
        compile_mode="max-autotune",
        gpu_pinned=True,
        fused_adamw=True,
        description="fp16 + compile + GPU pinned + fused AdamW + bs=4096",
    ),
]


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def make_model(device):
    torch.manual_seed(42)
    return LegNet(in_channels=4, task_mode="k562").to(device)


def train_epoch(model, data_iter, optimizer, scaler, scheduler, cfg: R2Cfg, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = cfg.amp_dtype != "none"
    accum = cfg.grad_accum

    for step, (x, y) in enumerate(data_iter):
        if not cfg.gpu_pinned:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        if use_amp:
            with autocast("cuda", dtype=amp_dtype):
                pred = model(x)
                loss = F.mse_loss(pred, y)
                if accum > 1:
                    loss = loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        else:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if accum > 1:
                loss = loss / accum
            loss.backward()
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item() * (accum if accum > 1 else 1)
        n_batches += 1

    return total_loss / max(n_batches, 1)


def eval_model(model, x_val, y_val, batch_size, gpu_pinned, device):
    model.eval()
    all_preds = []
    n = x_val.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x_val[i : i + batch_size]
            if not gpu_pinned:
                xb = xb.to(device)
            all_preds.append(model(xb).cpu())
    preds = torch.cat(all_preds).numpy().reshape(-1)
    tgts = y_val.cpu().numpy().reshape(-1)
    r, _ = pearsonr(preds, tgts)
    return float(r)


def run_config(cfg: R2Cfg, x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, device, epochs):
    print(f"\n{'=' * 72}")
    print(f"[{cfg.name}] {cfg.description}")
    print(
        f"  bs={cfg.batch_size}, compile={cfg.compile_mode}, fullgraph={cfg.compile_fullgraph}, "
        f"dynamic={cfg.compile_dynamic}"
    )
    print(
        f"  amp={cfg.amp_dtype}, fused_adamw={cfg.fused_adamw}, "
        f"gpu_pinned={cfg.gpu_pinned}, grad_accum={cfg.grad_accum}"
    )

    model = make_model(device)
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
        model = torch.compile(
            model,
            mode=cfg.compile_mode,
            fullgraph=cfg.compile_fullgraph,
            dynamic=cfg.compile_dynamic if cfg.compile_dynamic else None,
        )
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
    # For grad accumulation, optimizer steps happen every `grad_accum` micro-steps
    optimizer_steps_per_epoch = n_batches_per_epoch // max(cfg.grad_accum, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=optimizer_steps_per_epoch * epochs,
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

        val_r = eval_model(model, x_vl, y_vl, cfg.batch_size, cfg.gpu_pinned, device)
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
