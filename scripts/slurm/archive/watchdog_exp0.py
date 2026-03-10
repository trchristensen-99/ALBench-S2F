#!/usr/bin/env python3
"""
Watchdog script for Exp0 on CSHL HPC.
Resubmits the Slurm array job if it finishes/times out but results are missing.
Relying on the specific resume logic in exp0_yeast_scaling.py to skip completed work.
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM_SCRIPT = REPO_ROOT / "scripts" / "slurm" / "exp0_yeast_scaling.sh"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "exp0_yeast_scaling"
FRACTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_CHECK_INTERVAL_SEC = 1800  # 30 minutes
SLURM_BIN_CANDIDATE = Path("/cm/shared/apps/slurm/current/bin")


def _have_cmd(name: str) -> bool:
    """Return True if command is available in PATH."""
    return shutil.which(name) is not None


def _slurm_cmd(name: str) -> str | None:
    """Resolve slurm command path in PATH or common HPC install dir."""
    from_path = shutil.which(name)
    if from_path:
        return from_path
    candidate = SLURM_BIN_CANDIDATE / name
    if candidate.exists():
        return str(candidate)
    return None


def is_job_running(job_name: str = "exp0_yeast") -> bool:
    """Check if a job with the given name is running/pending using squeue."""
    squeue = _slurm_cmd("squeue")
    if squeue is None:
        print("squeue not found in PATH; treating scheduler state as unknown.")
        return False
    try:
        user = subprocess.check_output(["whoami"], text=True).strip()
        output = subprocess.check_output(
            [squeue, "-u", user, "--format=%j"], text=True, cwd=str(REPO_ROOT)
        )
        return job_name in output
    except Exception as exc:  # pragma: no cover - defensive shell interaction
        print(f"Error checking squeue: {exc}")
        return False


def is_local_process_running(pattern: str = "exp0_yeast_scaling.py") -> bool:
    """Fallback check: detect local python process if no scheduler is visible."""
    try:
        user = subprocess.check_output(["whoami"], text=True).strip()
        output = subprocess.check_output(
            ["ps", "-u", user, "-o", "pid=,cmd="], text=True, cwd=str(REPO_ROOT)
        )
        return any(pattern in line for line in output.splitlines())
    except Exception as exc:  # pragma: no cover
        print(f"Error checking local processes: {exc}")
        return False


def get_missing_fractions(output_dir: Path) -> list[float]:
    """Check which fractions are missing result.json."""
    completed: set[float] = set()

    # 1) Legacy fixed output path (single-run mode)
    if output_dir.exists():
        for frac in FRACTIONS:
            result_file = output_dir / f"fraction_{frac:.4f}" / "result.json"
            if result_file.exists():
                completed.add(frac)

    # 2) Timestamped run directories used by exp0_yeast_scaling.py
    if output_dir.exists():
        for run_dir in output_dir.glob("run_*"):
            if not run_dir.is_dir():
                continue
            for frac in FRACTIONS:
                result_file = run_dir / f"fraction_{frac:.4f}" / "result.json"
                if result_file.exists():
                    completed.add(frac)

    return [frac for frac in FRACTIONS if frac not in completed]


def submit_job(slurm_script: Path) -> None:
    """Submit the Slurm script from repository root."""
    sbatch = _slurm_cmd("sbatch")
    if sbatch is None:
        raise RuntimeError("sbatch not found in PATH or expected Slurm bin directory")
    if not slurm_script.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")
    print(f"Submitting job: {slurm_script}")
    output = subprocess.check_output(
        [sbatch, str(slurm_script)], text=True, cwd=str(REPO_ROOT)
    ).strip()
    print(f"Submission output: {output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", default="exp0_yeast")
    parser.add_argument("--once", action="store_true", help="Run one check cycle and exit.")
    parser.add_argument(
        "--check-interval-sec",
        type=int,
        default=DEFAULT_CHECK_INTERVAL_SEC,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT),
        help="Output directory root for exp0 yeast runs.",
    )
    return parser.parse_args()


def run_once(job_name: str, output_dir: Path) -> bool:
    """Run one watchdog cycle.

    Returns:
        True if all fractions are complete.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    missing = get_missing_fractions(output_dir=output_dir)
    if not missing:
        print(f"[{timestamp}] All fractions complete.")
        return True

    print(f"[{timestamp}] Missing fractions: {missing}")
    scheduler_running = is_job_running(job_name=job_name)
    local_running = is_local_process_running()
    if scheduler_running or local_running:
        print(
            f"[{timestamp}] Work appears to be running (scheduler={scheduler_running}, local={local_running})."
        )
        return False

    print(f"[{timestamp}] No active run detected. Submitting array job...")
    submit_job(SLURM_SCRIPT)
    return False


def main() -> None:
    """Entry point."""
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    print("Starting Exp0 Watchdog...")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Slurm script: {SLURM_SCRIPT}")
    print(f"Monitoring output dir: {output_dir}")

    if args.once:
        done = run_once(job_name=args.job_name, output_dir=output_dir)
        if done:
            print("Watchdog check completed: all done.")
        return

    while True:
        done = run_once(job_name=args.job_name, output_dir=output_dir)
        if done:
            print("All fractions complete. Watchdog exiting.")
            return
        print(f"Sleeping for {args.check_interval_sec} seconds...")
        time.sleep(args.check_interval_sec)


if __name__ == "__main__":
    main()
