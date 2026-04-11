#!/usr/bin/env python
"""Benchmark LegNet training speed with various optimization strategies.

Tests multiple approaches to speed up LegNet training on H100 GPUs.
Measures wall time per epoch and final val Pearson r for each approach.

Usage:
    uv run --no-sync python scripts/benchmark_legnet_speed.py --n-train 296000 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from torch.amp import GradScaler, autocast  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from models.legnet import LegNet, one_hot_encode_batch  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_k562_data(n_train: int, seed: int = 42):
    """Load K562 data: sequences from K562Dataset, labels from oracle pseudolabels."""
    from data.k562 import K562Dataset

    # Load train sequences and oracle labels
    ds = K562Dataset(
        data_path=str(REPO / "data" / "k562"),
        split="train",
        label_column="K562_log2FC",
        use_hashfrag=True,
    )
    all_seqs = list(ds.sequences)

    # Use oracle pseudolabels if available, otherwise ground truth
    oracle_path = REPO / "outputs" / "oracle_pseudolabels_k562_ag" / "train_oracle_labels.npz"
    if oracle_path.exists():
        oracle_data = np.load(str(oracle_path))
        all_labels = oracle_data["oracle_mean"].astype(np.float32)
        print(f"Using oracle pseudolabels ({len(all_labels)} labels)")
    else:
        all_labels = ds.labels.astype(np.float32)
        print(f"Using ground truth labels ({len(all_labels)} labels)")

    assert len(all_seqs) == len(all_labels), (
        f"Mismatch: {len(all_seqs)} seqs vs {len(all_labels)} labels"
    )

    # Subsample training set
    rng = np.random.RandomState(seed)
    n_total = len(all_seqs)
    if n_train > n_total:
        n_train = n_total
        print(f"Clamped to {n_total} available sequences")

    idx = rng.permutation(n_total)
    train_idx = idx[:n_train]

    train_seqs = [all_seqs[i] for i in train_idx]
    train_labels = all_labels[train_idx]

    # Load validation set
    val_ds = K562Dataset(
        data_path=str(REPO / "data" / "k562"),
        split="val",
        label_column="K562_log2FC",
        use_hashfrag=True,
    )
    val_seqs = list(val_ds.sequences)

    val_oracle_path = REPO / "outputs" / "oracle_pseudolabels_k562_ag" / "val_oracle_labels.npz"
    if val_oracle_path.exists():
        val_oracle = np.load(str(val_oracle_path))
        val_labels = val_oracle["oracle_mean"].astype(np.float32)
    else:
        val_labels = val_ds.labels.astype(np.float32)

    return train_seqs, train_labels, val_seqs, val_labels


def encode_sequences(sequences: list[str], seq_len: int = 200) -> torch.Tensor:
    """One-hot encode sequences to (N, 4, L) tensor."""
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
# Benchmark configurations
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    name: str
    batch_size: int = 512
    compile_mode: str | None = "max-autotune"  # None, "default", "reduce-overhead", "max-autotune"
    use_amp: bool = True
    gpu_pinned: bool = False  # Move all data to GPU at start
    grad_accum_steps: int = 1  # >1 for gradient accumulation
    skip_metrics: bool = False  # Skip per-batch metric collection
    use_cuda_graphs: bool = False
    optimizer_type: str = "adamw"  # "adamw" or "sgd"
    prefetch: bool = False  # Non-blocking transfers with prefetching
    num_workers: int = 0
    description: str = ""


BENCHMARKS = [
    # 0: Baseline (matches current production code)
    BenchConfig(
        name="baseline",
        batch_size=512,
        compile_mode="max-autotune",
        use_amp=True,
        gpu_pinned=False,
        description="Current production config (compile=max-autotune, amp, bs=512)",
    ),
    # 1: GPU-pinned data (avoid CPU->GPU per batch)
    BenchConfig(
        name="gpu_pinned",
        batch_size=512,
        compile_mode="max-autotune",
        use_amp=True,
        gpu_pinned=True,
        description="All data on GPU, no CPU->GPU transfer per batch",
    ),
    # 2: GPU-pinned + skip metric collection
    BenchConfig(
        name="gpu_pinned_nometrics",
        batch_size=512,
        compile_mode="max-autotune",
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        description="GPU-pinned + skip per-batch train metric collection",
    ),
    # 3: compile reduce-overhead (faster compile, may be faster for small models)
    BenchConfig(
        name="compile_reduce_overhead",
        batch_size=512,
        compile_mode="reduce-overhead",
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        description="compile reduce-overhead + gpu_pinned + skip metrics",
    ),
    # 4: No compile at all
    BenchConfig(
        name="no_compile",
        batch_size=512,
        compile_mode=None,
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        description="No torch.compile + gpu_pinned + skip metrics",
    ),
    # 5: Grad accumulation (effective bs=2048, forward bs=512)
    BenchConfig(
        name="grad_accum_4x",
        batch_size=512,
        compile_mode="max-autotune",
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        grad_accum_steps=4,
        description="Grad accum 4x (eff bs=2048), gpu_pinned, skip metrics",
    ),
    # 6: Large batch with grad accum (effective bs=4096)
    BenchConfig(
        name="grad_accum_8x",
        batch_size=512,
        compile_mode="max-autotune",
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        grad_accum_steps=8,
        description="Grad accum 8x (eff bs=4096), gpu_pinned, skip metrics",
    ),
    # 7: SGD with momentum (simpler optimizer)
    BenchConfig(
        name="sgd_momentum",
        batch_size=512,
        compile_mode="max-autotune",
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        optimizer_type="sgd",
        description="SGD+momentum instead of AdamW",
    ),
    # 8: Larger batch natively (bs=1024, no accum)
    BenchConfig(
        name="bs1024_native",
        batch_size=1024,
        compile_mode="max-autotune",
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        description="Native bs=1024, gpu_pinned, skip metrics",
    ),
    # 9: fp32 (no AMP) + gpu_pinned to check if AMP overhead exists
    BenchConfig(
        name="fp32_gpu_pinned",
        batch_size=512,
        compile_mode="max-autotune",
        use_amp=False,
        gpu_pinned=True,
        skip_metrics=True,
        description="fp32 (no AMP) + gpu_pinned + skip metrics",
    ),
    # 10: compile default mode (less aggressive)
    BenchConfig(
        name="compile_default",
        batch_size=512,
        compile_mode="default",
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        description="compile mode=default + gpu_pinned + skip metrics",
    ),
    # 11: CUDA graphs
    BenchConfig(
        name="cuda_graphs",
        batch_size=512,
        compile_mode=None,  # CUDA graphs don't mix well with torch.compile
        use_amp=True,
        gpu_pinned=True,
        skip_metrics=True,
        use_cuda_graphs=True,
        description="CUDA graphs + gpu_pinned + skip metrics (no compile)",
    ),
]


# ---------------------------------------------------------------------------
# GPU-pinned DataLoader (all data on GPU)
# ---------------------------------------------------------------------------


class GPUTensorDataset:
    """Dataset that lives entirely on GPU. Returns slices directly."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.n = x.shape[0]

    def __len__(self):
        return self.n


class GPUBatchIterator:
    """Iterate over GPU-resident data in batches."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
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
            x = self.x
            y = self.y

        for i in range(0, self.n, self.batch_size):
            if self.drop_last and i + self.batch_size > self.n:
                break
            yield x[i : i + self.batch_size], y[i : i + self.batch_size]


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def train_epoch_benchmark(
    model: nn.Module,
    data_iter,
    optimizer,
    device: torch.device,
    cfg: BenchConfig,
    scaler: GradScaler | None = None,
) -> dict:
    """Train one epoch with benchmarked config."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds = [] if not cfg.skip_metrics else None
    all_tgts = [] if not cfg.skip_metrics else None

    for step, (x_batch, y_batch) in enumerate(data_iter):
        if not cfg.gpu_pinned:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

        if cfg.use_amp and scaler is not None:
            with autocast("cuda"):
                pred = model(x_batch)
                loss = F.mse_loss(pred, y_batch)
                if cfg.grad_accum_steps > 1:
                    loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            if cfg.grad_accum_steps > 1:
                loss = loss / cfg.grad_accum_steps

            loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * (cfg.grad_accum_steps if cfg.grad_accum_steps > 1 else 1)
        n_batches += 1

        if not cfg.skip_metrics:
            all_preds.append(pred.detach())
            all_tgts.append(y_batch.detach())

    metrics = {"loss": total_loss / max(n_batches, 1)}
    if not cfg.skip_metrics and all_preds:
        preds = torch.cat(all_preds).cpu().numpy().reshape(-1)
        tgts = torch.cat(all_tgts).cpu().numpy().reshape(-1)
        r, _ = pearsonr(preds, tgts)
        metrics["train_pearson_r"] = float(r)

    return metrics


def eval_epoch(model: nn.Module, data_iter, device: torch.device, gpu_pinned: bool = False) -> dict:
    """Evaluate on validation set."""
    model.eval()
    all_preds = []
    all_tgts = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in data_iter:
            if not gpu_pinned:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(pred.cpu())
            all_tgts.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy().reshape(-1)
    tgts = torch.cat(all_tgts).numpy().reshape(-1)
    r, _ = pearsonr(preds, tgts)
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_pearson_r": float(r),
    }


def run_cuda_graph_benchmark(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: BenchConfig,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    """Run benchmark using CUDA graphs for the training step."""
    bs = cfg.batch_size
    n_train = x_train.shape[0]
    n_batches = n_train // bs

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    # Create static input/target buffers for CUDA graph
    static_x = torch.zeros(bs, 4, 200, device=device)
    static_y = torch.zeros(bs, device=device)

    # Warmup
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if cfg.use_amp:
        with autocast("cuda"):
            pred = model(static_x)
            loss = F.mse_loss(pred, static_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred = model(static_x)
        loss = F.mse_loss(pred, static_y)
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Capture CUDA graph
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        if cfg.use_amp:
            with autocast("cuda"):
                static_pred = model(static_x)
                static_loss = F.mse_loss(static_pred, static_y)
            scaler.scale(static_loss).backward()
        else:
            static_pred = model(static_x)
            static_loss = F.mse_loss(static_pred, static_y)
            static_loss.backward()

    print("  CUDA graph captured successfully")

    epoch_times = []
    val_metrics_list = []

    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()

        # Shuffle
        idx = torch.randperm(n_train, device=device)
        x_shuffled = x_train[idx]
        y_shuffled = y_train[idx]

        for i in range(n_batches):
            start = i * bs
            static_x.copy_(x_shuffled[start : start + bs])
            static_y.copy_(y_shuffled[start : start + bs])
            g.replay()
            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        epoch_times.append(t1 - t0)

        # Validate
        val_iter = GPUBatchIterator(x_val, y_val, bs, shuffle=False, drop_last=False)
        val_m = eval_epoch(model, val_iter, device, gpu_pinned=True)
        val_metrics_list.append(val_m)
        print(f"  Epoch {epoch + 1}/{epochs}: {t1 - t0:.2f}s, val_r={val_m['val_pearson_r']:.4f}")

    return {
        "epoch_times": epoch_times,
        "val_metrics": val_metrics_list,
        "mean_epoch_time": float(np.mean(epoch_times[1:])),  # skip first (compile)
        "final_val_r": val_metrics_list[-1]["val_pearson_r"],
    }


def run_benchmark(
    cfg: BenchConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    """Run a single benchmark configuration."""
    print(f"\n{'=' * 70}")
    print(f"Benchmark: {cfg.name}")
    print(f"  {cfg.description}")
    print(f"  bs={cfg.batch_size}, compile={cfg.compile_mode}, amp={cfg.use_amp}")
    print(f"  gpu_pinned={cfg.gpu_pinned}, grad_accum={cfg.grad_accum_steps}")
    print(f"  skip_metrics={cfg.skip_metrics}, cuda_graphs={cfg.use_cuda_graphs}")
    print(f"  optimizer={cfg.optimizer_type}")
    print(f"{'=' * 70}")

    # Fresh model for each benchmark
    torch.manual_seed(42)
    model = LegNet(in_channels=4, task_mode="k562").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Move data to GPU if gpu_pinned
    if cfg.gpu_pinned:
        x_tr_gpu = x_train.to(device)
        y_tr_gpu = y_train.to(device)
        x_vl_gpu = x_val.to(device)
        y_vl_gpu = y_val.to(device)
        torch.cuda.synchronize()
        print(f"  Data pinned to GPU: train={x_tr_gpu.shape}, val={x_vl_gpu.shape}")
    else:
        x_tr_gpu = x_train
        y_tr_gpu = y_train
        x_vl_gpu = x_val
        y_vl_gpu = y_val

    # CUDA graphs path
    if cfg.use_cuda_graphs:
        return run_cuda_graph_benchmark(
            model,
            x_tr_gpu,
            y_tr_gpu,
            x_vl_gpu,
            y_vl_gpu,
            cfg,
            device,
            epochs,
            lr,
        )

    # Compile model
    if cfg.compile_mode is not None:
        print(f"  Compiling model (mode={cfg.compile_mode})...")
        t_compile_start = time.perf_counter()
        model = torch.compile(model, mode=cfg.compile_mode)
        t_compile_end = time.perf_counter()
        print(f"  Compile call took {t_compile_end - t_compile_start:.1f}s")

    # Optimizer
    if cfg.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    # Build data iterators
    if cfg.gpu_pinned:

        def train_iter_fn():
            return GPUBatchIterator(
                x_tr_gpu, y_tr_gpu, cfg.batch_size, shuffle=True, drop_last=True
            )

        def val_iter_fn():
            return GPUBatchIterator(
                x_vl_gpu, y_vl_gpu, cfg.batch_size, shuffle=False, drop_last=False
            )

    else:
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        def train_iter_fn():
            return DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=cfg.num_workers,
                persistent_workers=cfg.num_workers > 0,
                drop_last=True,
            )

        def val_iter_fn():
            return DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=cfg.num_workers,
            )

    epoch_times = []
    val_metrics_list = []

    for epoch in range(epochs):
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        train_m = train_epoch_benchmark(model, train_iter_fn(), optimizer, device, cfg, scaler)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        epoch_times.append(t1 - t0)

        # Validate every epoch
        val_m = eval_epoch(model, val_iter_fn(), device, gpu_pinned=cfg.gpu_pinned)
        val_metrics_list.append(val_m)

        print(
            f"  Epoch {epoch + 1}/{epochs}: {t1 - t0:.2f}s, "
            f"train_loss={train_m['loss']:.4f}, "
            f"val_r={val_m['val_pearson_r']:.4f}"
        )

    return {
        "epoch_times": epoch_times,
        "val_metrics": val_metrics_list,
        "mean_epoch_time": float(np.mean(epoch_times[1:])),  # skip first (compile warmup)
        "first_epoch_time": float(epoch_times[0]),
        "final_val_r": val_metrics_list[-1]["val_pearson_r"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark LegNet training speed")
    parser.add_argument("--n-train", type=int, default=296000, help="Number of training sequences")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per benchmark")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="*",
        default=None,
        help="Names of benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO / "outputs" / "legnet_speed_benchmark"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")

    # Load data
    print(f"\nLoading data (n_train={args.n_train})...")
    train_seqs, train_labels, val_seqs, val_labels = load_k562_data(args.n_train)
    print(f"  Train: {len(train_seqs)}, Val: {len(val_seqs)}")

    # Encode
    print("Encoding sequences...")
    t0 = time.perf_counter()
    x_train = encode_sequences(train_seqs)
    y_train = torch.from_numpy(train_labels)
    x_val = encode_sequences(val_seqs)
    y_val = torch.from_numpy(val_labels)
    t1 = time.perf_counter()
    print(f"  Encoding took {t1 - t0:.1f}s")
    print(f"  x_train: {x_train.shape}, x_val: {x_val.shape}")

    # Select benchmarks
    if args.benchmarks:
        configs = [c for c in BENCHMARKS if c.name in args.benchmarks]
    else:
        configs = BENCHMARKS

    # Run benchmarks
    results = {}
    for cfg in configs:
        try:
            result = run_benchmark(
                cfg, x_train, y_train, x_val, y_val, device, args.epochs, args.lr
            )
            results[cfg.name] = {
                "config": {
                    "batch_size": cfg.batch_size,
                    "compile_mode": cfg.compile_mode,
                    "use_amp": cfg.use_amp,
                    "gpu_pinned": cfg.gpu_pinned,
                    "grad_accum_steps": cfg.grad_accum_steps,
                    "skip_metrics": cfg.skip_metrics,
                    "use_cuda_graphs": cfg.use_cuda_graphs,
                    "optimizer_type": cfg.optimizer_type,
                },
                "mean_epoch_time": result["mean_epoch_time"],
                "first_epoch_time": result.get("first_epoch_time"),
                "final_val_r": result["final_val_r"],
                "epoch_times": result["epoch_times"],
                "val_r_history": [m["val_pearson_r"] for m in result["val_metrics"]],
            }
            # Clear GPU cache between runs
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback

            traceback.print_exc()
            results[cfg.name] = {"error": str(e)}

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<30s} {'Mean s/epoch':>12s} {'First s/epoch':>13s} {'Val R':>8s}")
    print("-" * 70)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<30s} {'ERROR':>12s}")
        else:
            first = f"{r['first_epoch_time']:.1f}" if r.get("first_epoch_time") else "N/A"
            print(
                f"{name:<30s} {r['mean_epoch_time']:>12.2f} {first:>13s} {r['final_val_r']:>8.4f}"
            )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
