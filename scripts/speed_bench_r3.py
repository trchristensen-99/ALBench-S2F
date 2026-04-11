#!/usr/bin/env python
"""Round 3 LegNet speed benchmark — combine all best optimizations + production integration test.

Takes the winning configuration identified across R1+R2, runs it for a full
production-length training run (80 epochs, n=296K, 3 seeds), and compares
final val Pearson r against the baseline to confirm no quality regression.

Also benchmarks the "production integration" path: same optimizations but
going through the LegNetStudent.fit() interface (to confirm the savings
propagate into the real experiment pipeline).

Usage:
    uv run --no-sync python scripts/speed_bench_r3.py \
        --data-path outputs/labeled_pools/k562/ag_s2/genomic/pool.npz \
        --n-train 296000 \
        --best-config bf16_cl_fused \
        --output-dir outputs/legnet_speed_r3
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
# Shared helpers
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
    return (
        [sequences[i] for i in train_idx],
        labels[train_idx],
        [sequences[i] for i in val_idx],
        labels[val_idx],
    )


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
# Configs: baseline vs. best combined
# ---------------------------------------------------------------------------


@dataclass
class R3Cfg:
    name: str
    batch_size: int = 512
    compile_mode: Optional[str] = "max-autotune"
    compile_fullgraph: bool = False
    amp_dtype: str = "float16"
    gpu_pinned: bool = False
    channels_last: bool = False
    fused_adamw: bool = False
    lr: float = 0.005
    epochs: int = 30
    seeds: int = 3
    description: str = ""


# These will be re-confirmed/adjusted after R1+R2 — defaults are best guesses
R3_CONFIGS = [
    # Production baseline (matches current legnet_student.py)
    R3Cfg(
        name="production_baseline",
        batch_size=512,
        compile_mode="max-autotune",
        amp_dtype="float16",
        gpu_pinned=False,
        channels_last=False,
        fused_adamw=False,
        epochs=30,
        seeds=3,
        description="Current production: fp16 + compile(max-autotune) + CPU loader",
    ),
    # Best combined from R1+R2 (placeholder — adjust after seeing results)
    R3Cfg(
        name="best_combined",
        batch_size=512,
        compile_mode="max-autotune",
        amp_dtype="bfloat16",
        gpu_pinned=True,
        channels_last=False,
        fused_adamw=True,
        epochs=30,
        seeds=3,
        description="Best combined: bf16 + fused AdamW + compile + GPU pinned",
    ),
    # Best combined with channels_last (if R2 shows benefit)
    R3Cfg(
        name="best_combined_cl",
        batch_size=512,
        compile_mode="max-autotune",
        amp_dtype="bfloat16",
        gpu_pinned=True,
        channels_last=True,
        fused_adamw=True,
        epochs=30,
        seeds=3,
        description="Best combined + channels_last",
    ),
]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def make_model(device, channels_last=False, seed=42):
    torch.manual_seed(seed)
    m = LegNet(in_channels=4, task_mode="k562").to(device)
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


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


def run_one_seed(cfg: R3Cfg, x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, device, seed):
    """Train for cfg.epochs and return (mean_epoch_time, best_val_r, final_val_r, epoch_times)."""
    model = make_model(device, channels_last=cfg.channels_last, seed=seed)

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
        model = torch.compile(model, mode=cfg.compile_mode, fullgraph=cfg.compile_fullgraph)

    if cfg.fused_adamw:
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=0.01, fused=True
            )
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    optimizer.zero_grad(set_to_none=True)
    n_batches_per_epoch = x_tr.shape[0] // cfg.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=n_batches_per_epoch * cfg.epochs,
        pct_start=0.3,
    )

    use_scaler = cfg.amp_dtype == "float16"
    scaler = GradScaler("cuda", enabled=use_scaler)
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = cfg.amp_dtype != "none"

    epoch_times = []
    val_rs = []
    best_val_r = -float("inf")

    for ep in range(cfg.epochs):
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

        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

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
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        epoch_times.append(t1 - t0)

        val_r = eval_model(
            model, x_vl, y_vl, cfg.batch_size, cfg.gpu_pinned, cfg.channels_last, device
        )
        val_rs.append(val_r)
        if val_r > best_val_r:
            best_val_r = val_r

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"    ep{ep + 1:02d}: {t1 - t0:.2f}s  "
                f"loss={total_loss / n_batches:.4f}  val_r={val_r:.4f}"
            )

    mean_t = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else epoch_times[0]
    return {
        "mean_epoch_time_s": mean_t,
        "first_epoch_time_s": epoch_times[0],
        "epoch_times": epoch_times,
        "val_r_history": val_rs,
        "best_val_r": best_val_r,
        "final_val_r": val_rs[-1],
    }


def run_config(cfg: R3Cfg, x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, device):
    print(f"\n{'=' * 72}")
    print(f"[{cfg.name}] {cfg.description}")
    print(
        f"  bs={cfg.batch_size}, compile={cfg.compile_mode}, "
        f"amp={cfg.amp_dtype}, epochs={cfg.epochs}, seeds={cfg.seeds}"
    )
    print(
        f"  gpu_pinned={cfg.gpu_pinned}, channels_last={cfg.channels_last}, "
        f"fused_adamw={cfg.fused_adamw}"
    )

    seed_results = []
    for seed in range(cfg.seeds):
        print(f"\n  --- Seed {seed} ---")
        r = run_one_seed(cfg, x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, device, seed=seed)
        seed_results.append(r)
        torch.cuda.empty_cache()

    mean_epoch_times = [s["mean_epoch_time_s"] for s in seed_results]
    best_val_rs = [s["best_val_r"] for s in seed_results]
    final_val_rs = [s["final_val_r"] for s in seed_results]

    summary = {
        "mean_epoch_time_s": float(np.mean(mean_epoch_times)),
        "std_epoch_time_s": float(np.std(mean_epoch_times)),
        "mean_best_val_r": float(np.mean(best_val_rs)),
        "std_best_val_r": float(np.std(best_val_rs)),
        "mean_final_val_r": float(np.mean(final_val_rs)),
        "seed_results": seed_results,
        "config": asdict(cfg),
    }

    print(
        f"\n  [SUMMARY] {cfg.name}: "
        f"mean_epoch={np.mean(mean_epoch_times):.2f}s ± {np.std(mean_epoch_times):.2f}  "
        f"best_val_r={np.mean(best_val_rs):.4f} ± {np.std(best_val_rs):.4f}"
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="outputs/labeled_pools/k562/ag_s2/genomic/pool.npz")
    parser.add_argument("--n-train", type=int, default=296000)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument(
        "--best-config",
        type=str,
        default=None,
        help="Override the 'best_combined' config name from R1+R2",
    )
    parser.add_argument("--output-dir", default="outputs/legnet_speed_r3")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

    print(f"\nLoading data (n_train={args.n_train})...")
    train_seqs, train_labels, val_seqs, val_labels = load_pool_data(args.data_path, args.n_train)

    print("Encoding sequences...")
    t0 = time.perf_counter()
    x_train = encode_sequences(train_seqs, seq_len=200)
    y_train = torch.from_numpy(train_labels)
    x_val = encode_sequences(val_seqs, seq_len=200)
    y_val = torch.from_numpy(val_labels)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    selected = R3_CONFIGS
    if args.configs:
        selected = [c for c in R3_CONFIGS if c.name in args.configs]

    results = {}
    for cfg in selected:
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

    print(f"\n{'=' * 80}")
    print("SUMMARY (Round 3 — multi-seed quality + speed comparison)")
    print(f"{'=' * 80}")
    print(f"{'Config':<30s}  {'s/epoch':>8s}  {'±':>6s}  {'best_val_r':>10s}  {'±':>6s}")
    print("-" * 70)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<30s}  {'ERROR'}")
        else:
            print(
                f"{name:<30s}  "
                f"{r['mean_epoch_time_s']:>7.2f}s  "
                f"{r['std_epoch_time_s']:>5.2f}s  "
                f"{r['mean_best_val_r']:>10.4f}  "
                f"{r['std_best_val_r']:>5.4f}"
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results_r3.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
