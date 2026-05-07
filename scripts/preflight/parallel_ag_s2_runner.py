"""Parallel multi-AG-S2 trainer for the K562 oracle.

AlphaGenome encoder is ~91M params + heads ~5M, much larger than LegNet.
With JAX preallocation disabled and MEM_FRACTION=0.4, 2 instances fit
comfortably on an H100 (each gets ~32GB). 3 instances may fit but
risks OOM during peak memory; 2× is the safe default.

JAX-specific env required for multi-process GPU sharing:
    XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't claim all GPU at start
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4    # cap each process at 40%
    XLA_PYTHON_CLIENT_ALLOCATOR=platform  # finer-grained alloc

Usage:
    uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py \\
        configs.json [k_parallel=2]

configs.json format (one per debias config):
    [
      {"label": "...", "fold_id": 0, "n_folds": 10,
       "stage1_dir": "/path/to/oracle_0",
       "output_dir": "outputs/.../candidate/fold_0",
       "extra_overrides": ["++neg_fraction=0.05", "++debias_mode=cpg_invariance"]},
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
    k_parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_gpus = int(os.environ.get("N_GPUS", "1"))
    gpu_iter = itertools.cycle(range(n_gpus)) if n_gpus > 1 else None
    configs = json.loads(cfg_path.read_text())
    print(f"Loaded {len(configs)} AG-S2 configs, k_parallel={k_parallel}, N_GPUS={n_gpus}")
    pending = list(configs)
    running: list[tuple[subprocess.Popen, dict, object]] = []
    completed: list[dict] = []
    failed: list[tuple[dict, int]] = []
    t_start = time.time()
    while pending or running:
        while len(running) < k_parallel and pending:
            cfg = pending.pop(0)
            label = cfg.get("label", "?")
            out_dir = Path(cfg["output_dir"])
            if not out_dir.is_absolute():
                out_dir = REPO / out_dir
            if (out_dir / "test_metrics.json").exists():
                print(f"  [skip] {label} (already done)")
                completed.append(cfg)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "uv",
                "run",
                "--no-sync",
                "python",
                str(REPO / "experiments/train_stage2_k562_hashfrag.py"),
                "--config-name",
                "stage2_k562_oracle",
                f"++fold_id={cfg.get('fold_id', 0)}",
                f"++n_folds={cfg.get('n_folds', 10)}",
                f"++stage1_dir={cfg['stage1_dir']}",
                f"++output_dir={out_dir}",
                "++use_full_dataset=True",
                f"++epochs={cfg.get('epochs', 80)}",
                f"++early_stop_patience={cfg.get('patience', 15)}",
                "++wandb_mode=online",
            ]
            cmd.extend(cfg.get("extra_overrides", []))
            log_path = out_dir / "stdout.log"
            log_f = open(log_path, "w")
            env = os.environ.copy()
            # Critical: enable multi-process JAX GPU sharing
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"  # 40% of GPU per process
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
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
        time.sleep(15)

    elapsed = time.time() - t_start
    print(f"\nTotal wall: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  completed: {len(completed)}/{len(configs)}")
    print(f"  failed:    {len(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
