#!/usr/bin/env python3
"""
Watchdog script for Exp0 on CSHL HPC.
Resubmits the Slurm array job if it finishes/times out but results are missing.
Relying on the specific resume logic in exp0_yeast_scaling.py to skip completed work.
"""

import glob
import subprocess
import sys
import time
from pathlib import Path

# Config
SLURM_SCRIPT = "scripts/slurm/exp0_yeast_scaling.sh"
OUTPUT_DIR = "outputs/exp0_yeast_scaling/seed_42"
FRACTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
CHECK_INTERVAL_SEC = 1800  # 30 minutes


def is_job_running(job_name="exp0_yeast"):
    """Check if a job with the given name is running/pending using squeue."""
    try:
        # Check all jobs for the current user
        user = subprocess.check_output("whoami", shell=True).strip().decode()
        output = subprocess.check_output(f"squeue -u {user} --format=%j", shell=True).decode()
        return job_name in output
    except Exception as e:
        print(f"Error checking squeue: {e}")
        return True  # Assume running to be safe


def get_missing_fractions():
    """Check which fractions are missing result.json."""
    missing = []
    base_path = Path(OUTPUT_DIR)

    if not base_path.exists():
        return FRACTIONS

    for frac in FRACTIONS:
        # Format matching the script: f"fraction_{frac:.4f}"
        frac_dir = base_path / f"fraction_{frac:.4f}"
        result_file = frac_dir / "result.json"

        if not result_file.exists():
            missing.append(frac)

    return missing


def submit_job():
    """Submit the Slurm script."""
    print(f"Submitting job: {SLURM_SCRIPT}")
    try:
        output = subprocess.check_output(f"sbatch {SLURM_SCRIPT}", shell=True).decode()
        print(f"Submission output: {output.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Submission failed: {e}")


def main():
    print("Starting Exp0 Watchdog...")
    print(f"Monitoring output dir: {OUTPUT_DIR}")

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Check completion
        missing = get_missing_fractions()
        if not missing:
            print(f"[{timestamp}] ✅ All fractions complete! Watchdog exiting.")
            return

        print(f"[{timestamp}] Missing fractions: {missing}")

        # 2. Check if running
        if is_job_running():
            print(f"[{timestamp}] Job currently running. Waiting...")
        else:
            print(f"[{timestamp}] Job NOT running and work remains. Resubmitting...")
            submit_job()

        # 3. Sleep
        print(f"[{timestamp}] Sleeping for {CHECK_INTERVAL_SEC} seconds...")
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
