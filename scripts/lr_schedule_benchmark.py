#!/usr/bin/env python
"""LR schedule benchmark for LegNet — convergence speed comparison.

Tests 4 schedules at n=296K to find which converges fastest (measured by
val Pearson r at epochs 10, 20, 30, 40).

Schedules tested:
  baseline         : OneCycleLR max_lr=0.005, 80 epochs (current production)
  onecycle_fast    : OneCycleLR max_lr=0.003, 30 epochs, pct_start=0.3
  cosine_restarts  : CosineAnnealingWarmRestarts T_0=10, T_mult=2, eta_min=1e-5
  aggressive_decay : lr=0.01, ReduceLROnPlateau factor=0.5, patience=3
  warmup_cosine    : 5-epoch linear warmup to lr=0.005, then cosine decay to 0

All configs use: bf16 AMP, GPU-pinned data, fused AdamW, compile(max-autotune).
Best 1-2 schedules from Round 1 are refined in Round 2 (configurable via --round).

Usage (SLURM):
    uv run --no-sync python scripts/lr_schedule_benchmark.py \\
        --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \\
        --n-train 296382 --round 1 \\
        --output-dir outputs/lr_schedule_benchmark

Round 2 (refine best):
    uv run --no-sync python scripts/lr_schedule_benchmark.py \\
        --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \\
        --n-train 296382 --round 2 \\
        --output-dir outputs/lr_schedule_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from torch.amp import autocast  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from models.legnet import LegNet, one_hot_encode_batch  # noqa: E402  # isort: skip

# Epochs at which to snapshot val_r for convergence comparison
SNAPSHOT_EPOCHS = [10, 20, 30, 40]
MAX_EPOCHS = max(SNAPSHOT_EPOCHS)  # 40

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pool_data(pool_path: str, n_train: int, seed: int = 42):
    """Load pool.npz -> (train_seqs, train_labels, val_seqs, val_labels)."""
    p = Path(pool_path)
    if p.exists():
        print(f"Loading pool from {p}")
        data = np.load(p, allow_pickle=True)
        sequences = data["sequences"].tolist()
        labels = data["labels"].astype(np.float32)
    else:
        raise FileNotFoundError(
            f"Pool not found at {p}. "
            f"Generate with: python scripts/generate_labeled_pools.py "
            f"--task k562 --oracle ag_s2 --reservoir genomic"
        )

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
# GPU batch iterator (all data lives on GPU)
# ---------------------------------------------------------------------------


class GPUBatchIterator:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = x.shape[0]

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.x.device)
            x, y = self.x[idx], self.y[idx]
        else:
            x, y = self.x, self.y
        for i in range(0, self.n - self.batch_size + 1, self.batch_size):
            yield x[i : i + self.batch_size], y[i : i + self.batch_size]


# ---------------------------------------------------------------------------
# Schedule configs
# ---------------------------------------------------------------------------


@dataclass
class ScheduleConfig:
    name: str
    schedule_type: str  # "onecycle" | "cosine_restarts" | "plateau" | "warmup_cosine"
    # Common
    batch_size: int = 512
    max_epochs: int = MAX_EPOCHS
    weight_decay: float = 0.01
    # OneCycleLR params
    max_lr: float = 0.005
    pct_start: float = 0.3
    # CosineAnnealingWarmRestarts params
    base_lr: float = 0.001
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-5
    # ReduceLROnPlateau params
    plateau_lr: float = 0.01
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    plateau_min_lr: float = 1e-5
    # Warmup + cosine params
    warmup_epochs: int = 5
    warmup_target_lr: float = 0.005
    # Description
    description: str = ""
    seeds: int = 2
    # Additional refinement label
    round: int = 1


# ---------------------------------------------------------------------------
# Round 1: 5 schedules (baseline + 4 new)
# ---------------------------------------------------------------------------

ROUND1_CONFIGS = [
    ScheduleConfig(
        name="baseline_onecycle_80ep",
        schedule_type="onecycle",
        max_lr=0.005,
        pct_start=0.3,
        max_epochs=MAX_EPOCHS,  # only run 40 of the 80
        description="Current production: OneCycleLR max_lr=0.005 over 80ep (measured at ep10/20/30/40)",
        seeds=2,
        round=1,
    ),
    ScheduleConfig(
        name="onecycle_fast_30ep",
        schedule_type="onecycle",
        max_lr=0.003,
        pct_start=0.3,
        max_epochs=MAX_EPOCHS,
        description="OneCycleLR max_lr=0.003 over 30ep — fast convergence target",
        seeds=2,
        round=1,
    ),
    ScheduleConfig(
        name="cosine_restarts",
        schedule_type="cosine_restarts",
        base_lr=0.003,
        T_0=10,
        T_mult=2,
        eta_min=1e-5,
        max_epochs=MAX_EPOCHS,
        description="CosineAnnealingWarmRestarts T_0=10, T_mult=2, lr=0.003",
        seeds=2,
        round=1,
    ),
    ScheduleConfig(
        name="aggressive_decay",
        schedule_type="plateau",
        plateau_lr=0.01,
        plateau_factor=0.5,
        plateau_patience=3,
        plateau_min_lr=1e-5,
        max_epochs=MAX_EPOCHS,
        description="lr=0.01 + ReduceLROnPlateau factor=0.5 patience=3",
        seeds=2,
        round=1,
    ),
    ScheduleConfig(
        name="warmup_cosine",
        schedule_type="warmup_cosine",
        warmup_epochs=5,
        warmup_target_lr=0.005,
        max_epochs=MAX_EPOCHS,
        description="5-epoch linear warmup to lr=0.005 then cosine decay to 0",
        seeds=2,
        round=1,
    ),
]

# ---------------------------------------------------------------------------
# Round 2: refine the top-2 winners from Round 1 with tuned hyperparameters
# ---------------------------------------------------------------------------

ROUND2_CONFIGS = [
    # Refinement A: tighter pct_start sweep for OneCycleLR
    ScheduleConfig(
        name="onecycle_30ep_lr003_pct20",
        schedule_type="onecycle",
        max_lr=0.003,
        pct_start=0.2,
        max_epochs=MAX_EPOCHS,
        description="OneCycleLR max_lr=0.003, pct_start=0.2 (steeper ramp)",
        seeds=3,
        round=2,
    ),
    ScheduleConfig(
        name="onecycle_30ep_lr003_pct40",
        schedule_type="onecycle",
        max_lr=0.003,
        pct_start=0.4,
        max_epochs=MAX_EPOCHS,
        description="OneCycleLR max_lr=0.003, pct_start=0.4 (gentler ramp)",
        seeds=3,
        round=2,
    ),
    ScheduleConfig(
        name="onecycle_30ep_lr005_pct25",
        schedule_type="onecycle",
        max_lr=0.005,
        pct_start=0.25,
        max_epochs=MAX_EPOCHS,
        description="OneCycleLR max_lr=0.005 over 30ep, pct_start=0.25",
        seeds=3,
        round=2,
    ),
    # Refinement B: warmup+cosine with tighter params
    ScheduleConfig(
        name="warmup3_cosine_lr004",
        schedule_type="warmup_cosine",
        warmup_epochs=3,
        warmup_target_lr=0.004,
        max_epochs=MAX_EPOCHS,
        description="3-epoch warmup to lr=0.004 then cosine decay to 0",
        seeds=3,
        round=2,
    ),
    ScheduleConfig(
        name="warmup5_cosine_lr007",
        schedule_type="warmup_cosine",
        warmup_epochs=5,
        warmup_target_lr=0.007,
        max_epochs=MAX_EPOCHS,
        description="5-epoch warmup to lr=0.007 then cosine decay to 0",
        seeds=3,
        round=2,
    ),
    # Refinement C: cosine restarts with faster first cycle
    ScheduleConfig(
        name="cosine_restarts_T5_mult2",
        schedule_type="cosine_restarts",
        base_lr=0.003,
        T_0=5,
        T_mult=2,
        eta_min=1e-5,
        max_epochs=MAX_EPOCHS,
        description="CosineAnnealingWarmRestarts T_0=5, T_mult=2, lr=0.003",
        seeds=3,
        round=2,
    ),
]


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def make_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: ScheduleConfig,
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, Any]:
    """Create optimizer + scheduler for a given config.

    Returns (optimizer, scheduler). For plateau schedulers, the caller must
    call scheduler.step(val_loss) at the end of each epoch; for all others,
    scheduler.step() is called after every batch.
    """
    # Fused AdamW where available (H100 + PyTorch >= 2.0)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,  # will be overridden by scheduler
            weight_decay=cfg.weight_decay,
            fused=True,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=cfg.weight_decay,
        )

    total_steps = steps_per_epoch * cfg.max_epochs

    if cfg.schedule_type == "onecycle":
        # OneCycleLR: max_lr is the peak; initial lr = max_lr / div_factor (=25 by default)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.max_lr,
            total_steps=total_steps,
            pct_start=cfg.pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        step_per_batch = True

    elif cfg.schedule_type == "cosine_restarts":
        # Set initial lr
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.base_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.T_0 * steps_per_epoch,  # T_0 in steps (not epochs)
            T_mult=cfg.T_mult,
            eta_min=cfg.eta_min,
        )
        step_per_batch = True

    elif cfg.schedule_type == "plateau":
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.plateau_lr
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            min_lr=cfg.plateau_min_lr,
        )
        step_per_batch = False  # step per epoch with val_loss

    elif cfg.schedule_type == "warmup_cosine":
        # Linear warmup then cosine decay to 0
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.warmup_target_lr / max(cfg.warmup_epochs * steps_per_epoch, 1)

        warmup_steps = cfg.warmup_epochs * steps_per_epoch
        cosine_steps = total_steps - warmup_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear ramp from ~0 to warmup_target_lr
                return (current_step + 1) / max(warmup_steps, 1)
            else:
                # Cosine decay from warmup_target_lr to 0
                progress = (current_step - warmup_steps) / max(cosine_steps, 1)
                return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        # Fix the base lr so LambdaLR scales from the right value
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.warmup_target_lr
            pg["initial_lr"] = cfg.warmup_target_lr
        # Reinitialise so initial_lr is picked up
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        step_per_batch = True

    else:
        raise ValueError(f"Unknown schedule_type: {cfg.schedule_type!r}")

    return optimizer, scheduler, step_per_batch


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------


def run_one_seed(
    cfg: ScheduleConfig,
    x_train: torch.Tensor,  # GPU tensor
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict:
    """Train for cfg.max_epochs, snapshot val_r at SNAPSHOT_EPOCHS."""
    torch.manual_seed(seed)
    model = LegNet(in_channels=4, task_mode="k562").to(device)
    model = torch.compile(model, mode="max-autotune")

    steps_per_epoch = x_train.shape[0] // cfg.batch_size
    optimizer, scheduler, step_per_batch = make_optimizer_and_scheduler(model, cfg, steps_per_epoch)

    amp_dtype = torch.bfloat16

    epoch_times: list[float] = []
    val_r_history: list[float] = []
    snapshot_val_r: dict[int, float] = {}
    best_val_r = -float("inf")
    lr_history: list[float] = []

    for ep in range(cfg.max_epochs):
        model.train()
        t0 = time.perf_counter()
        data_iter = GPUBatchIterator(x_train, y_train, cfg.batch_size, shuffle=True)

        for x, y in data_iter:
            with autocast("cuda", dtype=amp_dtype):
                pred = model(x)
                loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step_per_batch:
                scheduler.step()

        torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(float(current_lr))

        # Evaluate
        val_r = _eval_pearson(model, x_val, y_val, cfg.batch_size, device, amp_dtype)
        val_r_history.append(val_r)
        if val_r > best_val_r:
            best_val_r = val_r

        epoch_no = ep + 1
        if epoch_no in SNAPSHOT_EPOCHS:
            snapshot_val_r[epoch_no] = val_r

        # ReduceLROnPlateau steps on val loss (need val_loss for that)
        if not step_per_batch:
            val_loss = _eval_loss(model, x_val, y_val, cfg.batch_size, device, amp_dtype)
            scheduler.step(val_loss)

        if epoch_no % 5 == 0 or epoch_no == 1:
            print(
                f"    ep{epoch_no:02d}: {epoch_times[-1]:.2f}s  "
                f"lr={current_lr:.2e}  val_r={val_r:.4f}"
            )

    mean_epoch_t = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else epoch_times[0]
    return {
        "mean_epoch_time_s": mean_epoch_t,
        "first_epoch_time_s": float(epoch_times[0]),
        "epoch_times": [float(t) for t in epoch_times],
        "val_r_history": [float(r) for r in val_r_history],
        "snapshot_val_r": {str(k): float(v) for k, v in snapshot_val_r.items()},
        "best_val_r": float(best_val_r),
        "final_val_r": float(val_r_history[-1]),
        "lr_history": lr_history,
    }


def _eval_pearson(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_val.shape[0], batch_size):
            xb = x_val[i : i + batch_size]
            with autocast("cuda", dtype=amp_dtype):
                p = model(xb).float()
            preds.append(p.cpu())
    preds_np = torch.cat(preds).numpy().reshape(-1)
    tgts_np = y_val.cpu().numpy().reshape(-1)
    r, _ = pearsonr(preds_np, tgts_np)
    return float(r)


def _eval_loss(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, x_val.shape[0], batch_size):
            xb = x_val[i : i + batch_size]
            yb = y_val[i : i + batch_size]
            with autocast("cuda", dtype=amp_dtype):
                p = model(xb)
                loss = F.mse_loss(p, yb)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Run a full config (multiple seeds)
# ---------------------------------------------------------------------------


def run_config(
    cfg: ScheduleConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
) -> dict:
    print(f"\n{'=' * 72}")
    print(f"[{cfg.name}] {cfg.description}")
    print(
        f"  schedule={cfg.schedule_type}, epochs={cfg.max_epochs}, seeds={cfg.seeds}, bs={cfg.batch_size}"
    )

    seed_results = []
    for seed in range(cfg.seeds):
        print(f"\n  --- Seed {seed} ---")
        r = run_one_seed(cfg, x_train, y_train, x_val, y_val, device, seed=seed)
        seed_results.append(r)
        torch.cuda.empty_cache()

    # Aggregate across seeds
    best_val_rs = [s["best_val_r"] for s in seed_results]
    mean_epoch_ts = [s["mean_epoch_time_s"] for s in seed_results]

    # Average snapshot val_r across seeds
    avg_snapshots: dict[str, float] = {}
    for ep_key in [str(e) for e in SNAPSHOT_EPOCHS]:
        vals = [s["snapshot_val_r"].get(ep_key, float("nan")) for s in seed_results]
        avg_snapshots[ep_key] = float(np.nanmean(vals))

    summary = {
        "mean_epoch_time_s": float(np.mean(mean_epoch_ts)),
        "std_epoch_time_s": float(np.std(mean_epoch_ts)),
        "mean_best_val_r": float(np.mean(best_val_rs)),
        "std_best_val_r": float(np.std(best_val_rs)),
        "avg_snapshot_val_r": avg_snapshots,  # {epoch: mean_val_r across seeds}
        "seed_results": seed_results,
        "config": asdict(cfg),
    }

    print(f"\n  [SUMMARY] {cfg.name}:")
    print(f"    mean_epoch_time: {np.mean(mean_epoch_ts):.2f}s ± {np.std(mean_epoch_ts):.2f}s")
    print(f"    best_val_r:      {np.mean(best_val_rs):.4f} ± {np.std(best_val_rs):.4f}")
    print(f"    val_r snapshots: " + " | ".join(f"ep{k}={v:.4f}" for k, v in avg_snapshots.items()))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LR schedule benchmark for LegNet")
    parser.add_argument(
        "--data-path",
        default="outputs/labeled_pools/k562/ag_s2/genomic/pool.npz",
        help="Path to pool.npz",
    )
    parser.add_argument("--n-train", type=int, default=296382, help="Training set size")
    parser.add_argument(
        "--round",
        type=int,
        choices=[1, 2],
        default=1,
        help="Round 1: initial sweep; Round 2: refine best schedule",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Subset of config names to run (default: all for the selected round)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lr_schedule_benchmark",
        help="Output directory",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

    # Select configs for this round
    all_cfgs = ROUND1_CONFIGS if args.round == 1 else ROUND2_CONFIGS
    if args.configs:
        all_cfgs = [c for c in all_cfgs if c.name in args.configs]

    print(f"\nRound {args.round}: running {len(all_cfgs)} config(s)")
    print(f"Max epochs: {MAX_EPOCHS}  Snapshots: {SNAPSHOT_EPOCHS}")
    print(f"n_train: {args.n_train}")

    # Load data
    print(f"\nLoading data from {args.data_path} ...")
    train_seqs, train_labels, val_seqs, val_labels = load_pool_data(args.data_path, args.n_train)

    print("Encoding sequences (200bp one-hot)...")
    t0 = time.perf_counter()
    x_train_cpu = encode_sequences(train_seqs, seq_len=200)
    y_train_cpu = torch.from_numpy(train_labels)
    x_val_cpu = encode_sequences(val_seqs, seq_len=200)
    y_val_cpu = torch.from_numpy(val_labels)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    print(f"  x_train: {x_train_cpu.shape}, x_val: {x_val_cpu.shape}")

    # Pin data to GPU
    print("Pinning data to GPU...")
    x_train = x_train_cpu.to(device)
    y_train = y_train_cpu.to(device)
    x_val = x_val_cpu.to(device)
    y_val = y_val_cpu.to(device)
    torch.cuda.synchronize()
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after data load: {mem_gb:.2f} GB")

    # Run all configs
    results: dict[str, dict] = {}
    for cfg in all_cfgs:
        try:
            r = run_config(cfg, x_train, y_train, x_val, y_val, device)
            results[cfg.name] = r
        except Exception as e:
            import traceback

            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[cfg.name] = {"error": str(e)}
        finally:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # --- Final summary table ---
    print(f"\n{'=' * 90}")
    print(f"ROUND {args.round} SUMMARY — val Pearson r @ ep10/20/30/40 + best + s/epoch")
    print(f"{'=' * 90}")
    col_w = 30
    ep_header = "  ".join(f"ep{e:<5d}" for e in SNAPSHOT_EPOCHS)
    print(f"{'Config':<{col_w}}  {ep_header}  {'best':>6s}  {'s/ep':>6s}")
    print("-" * 90)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<{col_w}}  ERROR: {r['error']}")
        else:
            snaps = r.get("avg_snapshot_val_r", {})
            ep_vals = "  ".join(f"{snaps.get(str(e), float('nan')):.4f}" for e in SNAPSHOT_EPOCHS)
            print(
                f"{name:<{col_w}}  {ep_vals}  "
                f"{r['mean_best_val_r']:>6.4f}  "
                f"{r['mean_epoch_time_s']:>5.1f}s"
            )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_round{args.round}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print recommendation
    _print_recommendation(results, args.round)


def _print_recommendation(results: dict, rnd: int) -> None:
    """Print which schedule to use / refine next."""
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        print("\nAll configs failed — no recommendation possible.")
        return

    # Rank by ep20 val_r (early convergence speed), then by ep40 best
    def score(item):
        snaps = item[1].get("avg_snapshot_val_r", {})
        ep20 = snaps.get("20", -1.0)
        ep40 = snaps.get("40", -1.0)
        return (ep20, ep40)

    ranked = sorted(valid.items(), key=score, reverse=True)

    print(f"\n--- Round {rnd} Recommendation ---")
    print("Ranked by ep20 val_r (then ep40):")
    for rank, (name, r) in enumerate(ranked, 1):
        snaps = r.get("avg_snapshot_val_r", {})
        ep20 = snaps.get("20", float("nan"))
        ep40 = snaps.get("40", float("nan"))
        print(
            f"  {rank}. {name}: ep20={ep20:.4f}  ep40={ep40:.4f}  best={r['mean_best_val_r']:.4f}"
        )

    winner = ranked[0][0]
    if rnd == 1:
        print(f"\nTop pick for Round 2 refinement: {winner}")
        print("Re-run with --round 2 after reviewing the results.")
    else:
        print(f"\nFinal recommended schedule: {winner}")
        print("Update LegNetStudent.TrainConfig with the corresponding params.")


if __name__ == "__main__":
    main()
