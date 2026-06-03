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


def _maybe_prebuild_cache(configs: list[dict]) -> Path | None:
    """If all configs share the same (d_train, seed, label_source, aug) tuple,
    pre-build the one-hot tensor cache ONCE before launching subprocess trials.
    All subprocesses share the cache; avoids encoding the dataset N times.

    Returns the cache_dir Path to pass to each subprocess (or None if heterogeneous).
    """
    if not configs:
        return None
    keys = set()
    for cfg in configs:
        keys.add(
            (
                cfg.get("d_train"),
                cfg.get("seed"),
                cfg.get("label_source", "ag_oracle"),
                # in_channels depends on arch — but we cache per-(payload_len, in_channels)
                # via the filename, so heterogeneous archs in one cell are OK if D/seed
                # match (different files written).
            )
        )
    if len(keys) > 1:
        print(f"  cache pre-build skipped (heterogeneous configs: {len(keys)} d_train/seed combos)")
        return None
    d_train, seed, label_source = next(iter(keys))
    cache_dir = REPO / "outputs" / "tensor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  pre-building tensor cache at {cache_dir}  (d={d_train}, seed={seed}, labels={label_source})"
    )
    # Run a quick subprocess to build the cache (one per in_channels/pad combo).
    # We only need to invoke load_data once; the easiest is a tiny inline script.
    # Use one_hot encoding at payload_len=200 for default in_channels=4 (LegNet) and =5
    # (DREAM-ATTN), and once with adapter padding if any config uses rc_shift.
    in_channels_set = set()
    pad_set = set()
    for cfg in configs:
        arch = cfg.get("arch", "legnet")
        in_channels_set.add(4 if arch == "legnet" else 5)
        pad_set.add(cfg.get("aug") in ("rc_shift", "rc_shift_evoaug"))
    # Pre-build all (in_channels, pad) combos
    for ic in in_channels_set:
        for pad in pad_set:
            cmd = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-c",
                (
                    "import sys; sys.path.insert(0, '.'); "
                    "from scripts.preflight.run_single import load_data; "
                    f"load_data({d_train}, {seed}, in_channels={ic}, "
                    f"label_source='{label_source}', pad_with_adapters={pad}, "
                    f"cache_dir='{cache_dir}')"
                ),
            ]
            t0 = time.time()
            ret = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            if ret.returncode == 0:
                print(f"    in_ch={ic} pad={pad}: built in {time.time() - t0:.1f}s")
            else:
                print(f"    in_ch={ic} pad={pad}: WARN build failed:\n{ret.stderr[-500:]}")
    return cache_dir


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

    # Pre-build the one-hot tensor cache once (saves ~30s × (k_parallel - 1) per batch)
    cache_dir = _maybe_prebuild_cache(configs)
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
                cfg.get("label_source", "ag_oracle"),
                "--output_dir",
                str(out_dir),
            ]
            # Optional EvoAug args — map int level 1/2/4 → light/medium/heavy
            # to match run_single.py choices (only int 0 = disabled / no flag).
            _ei = cfg.get("evoaug_intensity")
            if _ei and _ei not in (0, "0", None):
                if isinstance(_ei, int) or (isinstance(_ei, str) and _ei.isdigit()):
                    _ei = {1: "light", 2: "medium", 4: "heavy"}.get(int(_ei), "light")
                cmd.extend(["--evoaug_intensity", str(_ei)])
            if "evoaug_prob" in cfg:
                cmd.extend(["--evoaug_prob", str(cfg["evoaug_prob"])])
            # Optional chr-fold args (MPAC-style ensemble: each fold's val
            # chromosome is different; test_chrs stays as chr 7+13 for the SNV
            # eval). val_chrs / test_chrs are comma-sep strings.
            if cfg.get("val_chrs"):
                cmd.extend(["--val_chrs", str(cfg["val_chrs"])])
            if cfg.get("test_chrs"):
                cmd.extend(["--test_chrs", str(cfg["test_chrs"])])
            # Forward the pre-built tensor cache so trials skip one-hot encoding
            if cache_dir is not None:
                cmd.extend(["--cache_dir", str(cache_dir)])
            # HP_FAST=1 turns on all speedups (default for HP search; opt out via
            # HP_FAST=0 in env).
            if cfg.get("fast") or os.environ.get("HP_FAST", "1") == "1":
                cmd.append("--fast")
            else:
                if cfg.get("use_compile") or os.environ.get("USE_COMPILE") == "1":
                    cmd.append("--use_compile")
                if cfg.get("cudnn_benchmark") or os.environ.get("CUDNN_BENCHMARK") == "1":
                    cmd.append("--cudnn_benchmark")
                if cfg.get("eval_on_gpu") or os.environ.get("EVAL_ON_GPU") == "1":
                    cmd.append("--eval_on_gpu")
                ete = cfg.get("eval_test_every") or os.environ.get("EVAL_TEST_EVERY")
                if ete:
                    cmd.extend(["--eval_test_every", str(ete)])
                ebm = cfg.get("eval_batch_mult") or os.environ.get("EVAL_BATCH_MULT")
                if ebm:
                    cmd.extend(["--eval_batch_mult", str(ebm)])
            cmd.extend(["--hp", *cfg["hp_overrides"]])
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
                # NEW: classify failure by scanning last 50 stdout lines
                cause = "unknown"
                tail = ""
                try:
                    with open(log_path) as _lf:
                        tail = "".join(_lf.readlines()[-50:])
                    if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
                        cause = "oom"
                    elif "nan" in tail.lower() or "NaN" in tail:
                        cause = "nan"
                    elif ret == -9 or ret == 137 or "Killed" in tail:
                        cause = "sigkill"
                    elif "AssertionError" in tail or "ValueError" in tail:
                        cause = "assert"
                    elif ret == 1:
                        cause = "exit1"
                    else:
                        cause = f"exit{ret}"
                except Exception:
                    pass
                # Write failure marker
                try:
                    fmark = out_dir / "failure.json"
                    import json as _json
                    fmark.write_text(_json.dumps({
                        "label": label, "exit_code": ret, "cause": cause,
                        "tail": tail[-2000:],
                    }, indent=2))
                except Exception:
                    pass
                failed.append((cfg, ret, cause))
                print(f"  ✗ [FAILED +{elapsed:.0f}s] {label} (exit {ret}, cause={cause})")
        time.sleep(5)

    elapsed = time.time() - t_start
    print(f"\nTotal wall: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  completed: {len(completed)}/{len(configs)}")
    print(f"  failed:    {len(failed)}")
    if failed:
        # Tally failure causes
        cause_counts = {}
        for entry in failed:
            cause = entry[2] if len(entry) > 2 else "unknown"
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        print(f"  failure causes: {cause_counts}")
        for entry in failed:
            label = entry[0].get("label", "?")
            ret = entry[1]
            cause = entry[2] if len(entry) > 2 else "?"
            print(f"    - {label}: exit {ret} ({cause})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
