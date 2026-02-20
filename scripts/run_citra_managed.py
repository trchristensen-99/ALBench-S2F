#!/usr/bin/env python3
"""
Managed execution script for Citra (or any multi-GPU node).
Monitors GPU usage and schedules Exp0 fractions on available GPUs.
"""

import os
import shutil
import subprocess
import sys
import time

# Fractions to run (largest to smallest)
FRACTIONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
CMD_TEMPLATE = "bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction {} --wandb-mode offline --gpu {}"
LOG_DIR = "logs"


def get_free_gpus(threshold_mb=1000):
    """
    Get list of GPU indices with memory usage usage below threshold.
    Uses nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        free_gpus = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            idx, mem_used = line.split(",")
            if int(mem_used) < threshold_mb:
                free_gpus.append(int(idx))
        return free_gpus
    except Exception as e:
        print(f"Error checking GPUs: {e}")
        return []


def main():
    print("Starting managed execution for Exp0 (Yeast Scaling)...")

    # Queue of fractions to run
    queue = list(FRACTIONS)

    # Track running processes: {gpu_idx: (fraction, subprocess.Popen)}
    running = {}

    # Create logs dir
    os.makedirs(LOG_DIR, exist_ok=True)

    while queue or running:
        # 1. Check completed jobs
        completed_gpus = []
        for gpu, (frac, proc) in running.items():
            ret = proc.poll()
            if ret is not None:
                print(f"✅ Fraction {frac} on GPU {gpu} finished (exit code {ret})")
                completed_gpus.append(gpu)

        for gpu in completed_gpus:
            del running[gpu]

        # 2. Assign new jobs if allowed
        if queue:
            # Check actual hardware availability
            hardware_free = get_free_gpus()

            # Filter: must be hardware-free AND not currently tracked by us
            # (nvidia-smi might lag, so trust our tracking first)
            available = [g for g in hardware_free if g not in running]

            # Sort to prefer lower IDs? Doesn't matter.
            available.sort()

            while available and queue:
                gpu = available.pop(0)
                frac = queue.pop(0)

                log_file = f"{LOG_DIR}/exp0_yeast_{frac}.log"
                cmd = CMD_TEMPLATE.format(frac, gpu)

                print(f"🚀 Launching Fraction {frac} on GPU {gpu} (Logs: {log_file})")

                # Launch background process
                with open(log_file, "w") as f_out:
                    proc = subprocess.Popen(cmd, shell=True, stdout=f_out, stderr=subprocess.STDOUT)

                running[gpu] = (frac, proc)
                time.sleep(2)  # Stagger launches slightly

        # 3. Wait before next cycle
        if queue or running:
            time.sleep(10)

    print("All jobs managed and completed.")


if __name__ == "__main__":
    main()
