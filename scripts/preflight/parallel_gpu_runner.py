"""Parallel multi-model trainer with optional multi-GPU support.

LegNet (~2M params) and DREAM-RNN (~1M) are tiny; multiple instances
can train concurrently on one H100 (80 GB) without OOM.

Single GPU mode (default): all k_parallel subprocesses share GPU 0
via CUDA driver context multiplexing. Best when each model is small
and the GPU is underutilized by one process.

Multi-GPU mode (env N_GPUS>1): processes are round-robin assigned
to N_GPUS via CUDA_VISIBLE_DEVICES. Use when SLURM allocation has
multiple GPUs and you want isolated per-process CUDA contexts (less
contention but ties up more cluster slots).

Throughput vs serial:
- Single GPU, 6× LegNet: ~3-5x speedup vs serial
- Multi-GPU (N), N×6 LegNet: ~N× more throughput on top of that

Usage:
    # Single GPU
    uv run --no-sync python scripts/preflight/parallel_gpu_runner.py \\
        configs.json [k_parallel=6]
    # Multi-GPU (4 H100s, 6 models each = 24 concurrent)
    N_GPUS=4 uv run --no-sync python scripts/preflight/parallel_gpu_runner.py \\
        configs.json 24

configs.json format:
    [
      {"label": "...", "arch": "legnet", "d_train": 10000, "seed": 42,
       "epochs": 80, "patience": 15, "aug": "rev_complement",
       "output_dir": "...", "hp_overrides": ["lr=0.003", "batch_size=512"]},
      ...
    ]
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cfg_path = Path(sys.argv[1])
    k_parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_gpus = int(os.environ.get("N_GPUS", "1"))
    gpu_iter = itertools.cycle(range(n_gpus)) if n_gpus > 1 else None
    configs = json.loads(cfg_path.read_text())
    print(f"Loaded {len(configs)} configs, k_parallel={k_parallel}, N_GPUS={n_gpus}")
    pending = list(configs)
    running: list[tuple[subprocess.Popen, dict, object]] = []
    completed: list[dict] = []
    failed: list[tuple[dict, int]] = []
    t_start = time.time()
    while pending or running:
        # Launch new processes up to limit
        while len(running) < k_parallel and pending:
            cfg = pending.pop(0)
            label = cfg.get("label", "?")
            out_dir = Path(cfg["output_dir"])
            if not out_dir.is_absolute():
                out_dir = REPO / out_dir
            if (out_dir / "result.json").exists():
                print(f"  [skip] {label} (already done)")
                completed.append(cfg)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "uv",
                "run",
                "--no-sync",
                "python",
                str(REPO / "scripts/preflight/run_single.py"),
                "--arch",
                cfg["arch"],
                "--d_train",
                str(cfg["d_train"]),
                "--seed",
                str(cfg["seed"]),
                "--epochs",
                str(cfg.get("epochs", 80)),
                "--early_stop_patience",
                str(cfg.get("patience", 15)),
                "--augmentations",
                cfg["aug"],
                "--label_source",
                "ag_oracle",
                "--output_dir",
                str(out_dir),
                "--hp",
                *cfg["hp_overrides"],
            ]
            log_path = out_dir / "stdout.log"
            log_f = open(log_path, "w")
            env = os.environ.copy()
            gpu_tag = ""
            if gpu_iter is not None:
                gpu_idx = next(gpu_iter)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                gpu_tag = f" [GPU{gpu_idx}]"
            p = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env
            )
            running.append((p, cfg, log_f))
            elapsed = time.time() - t_start
            print(f"  [start +{elapsed:.0f}s]{gpu_tag} {label} (pid {p.pid})")
        # Poll for completed
        for entry in list(running):
            p, cfg, log_f = entry
            ret = p.poll()
            if ret is None:
                continue
            running.remove(entry)
            log_f.close()
            label = cfg.get("label", "?")
            elapsed = time.time() - t_start
            if ret == 0:
                completed.append(cfg)
                print(f"  ✓ [done +{elapsed:.0f}s] {label}")
            else:
                failed.append((cfg, ret))
                print(f"  ✗ [FAILED +{elapsed:.0f}s] {label} (exit {ret})")
        time.sleep(5)

    elapsed = time.time() - t_start
    print(f"\nTotal wall: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  completed: {len(completed)}/{len(configs)}")
    print(f"  failed:    {len(failed)}")
    if failed:
        for cfg, ret in failed:
            print(f"    - {cfg.get('label', '?')}: exit {ret}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
