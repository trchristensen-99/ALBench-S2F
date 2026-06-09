"""Submit the epoch-budget diagnostic: random HP coverage x reservoir x D, at
epochs=60/patience=10, logging best_epoch per run.

Cells = RESERVOIRS x DS. Each cell is a SLURM array (ARRAY_TASKS tasks) where each task
runs K random configs under its own hp_seed into its own seed subdir, so we get
ARRAY_TASKS*K configs/cell. Post-hoc: scripts/analysis/analyze_epoch_budget.py reads
every */d*/seed*/r00_random_*_meta.json and attributes best_epoch to the HP axes.

Run from the HPC login node:
  export PATH=/cm/shared/apps/slurm/current/bin:$PATH
  python scripts/submit_epoch_diagnostic.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"
SCRIPT = REPO / "scripts/slurm/epoch_diagnostic.sh"

RESERVOIRS = ["genomic", "random", "prm_10pct", "evoaug_heavy", "motif_planted_v2"]
DS = [30_000, 300_000]
ARRAY_TASKS = 5  # array=0-4
K = 5  # configs per task -> 25 configs/cell

# (qos, walltime) per D — bakeoff is held, so the default group limit is free.
QOS = {30_000: ("fast", "04:00:00"), 300_000: ("default", "12:00:00")}


def main():
    for r in RESERVOIRS:
        for d in DS:
            qos, wt = QOS[d]
            cmd = [
                SBATCH,
                "--parsable",
                f"--array=0-{ARRAY_TASKS - 1}",
                f"--qos={qos}",
                f"--time={wt}",
                f"--job-name=epdiag_{r}_d{d}",
                f"--export=ALL,DIAG_RESERVOIR={r},DIAG_D={d},DIAG_K={K}",
                str(SCRIPT),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                print(f"  {r:18s} d{d:<6d} [{qos}] -> {res.stdout.strip()}")
            else:
                print(f"  {r:18s} d{d:<6d} FAILED: {res.stderr.strip()[:160]}")


if __name__ == "__main__":
    main()
