"""Overnight reservoir-strategy scaling at high D (500k, 1M, 3M).

PI's preferred reservoir baselines: random, prm_1pct, prm_20pct, evoaug_heavy,
motif_planted (and genomic, which is capped at pool=315k → only D=300k here).
Generative strategies are unbounded in sequence count; we push to D=3M for
the cheapest ones.

Per cell: keep HP sweep ON (exp1_1_scaling default) but reduce ensemble +
replicates at high D so each job fits in walltime.

Usage: python scripts/scaling_hp_search/submit_overnight_reservoir_highd.py [--dry_run]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"

# Generative strategies (no pool size limit)
GENERATIVE = ["random", "prm_1pct", "prm_20pct", "evoaug_heavy", "motif_planted"]
CHEAP = ["random", "prm_1pct", "prm_20pct"]  # for D=3M; mutation+random are cheapest to generate

# (D, strategies, n_replicates, ensemble_size, epochs, walltime, mem_gb)
TIERS = [
    (300_000, ["genomic"], 3, 5, 30, "24:00:00", 100),  # full pool, biological reference
    (500_000, GENERATIVE, 2, 3, 30, "24:00:00", 80),
    (1_000_000, GENERATIVE, 2, 3, 25, "48:00:00", 100),
    (3_000_000, CHEAP, 2, 2, 20, "72:00:00", 120),
]


def slurm_text(d, strategy, n_reps, ens, epochs, walltime, mem_gb, label):
    out_dir = f"{REPO}/outputs/exp1_1/k562/legnet_highd_pi/{label}"
    return (
        f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time={walltime}
#SBATCH --mem={mem_gb}G
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD"
echo "=== Reservoir scaling: K562 LegNet AG_S2 D={d} strategy={strategy} ==="
uv run --no-sync python experiments/exp1_1_scaling.py \\
    --task k562 --student legnet --oracle ag_s2 \\
    --reservoir {strategy} \\
    --training-sizes {d} \\
    --n-replicates {n_reps} --ensemble-size {ens} \\
    --epochs {epochs} --early-stop-patience 8 \\
    --output-dir {out_dir}
""",
        out_dir,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    submitted = 0
    failed = 0
    for d, strategies, n_reps, ens, epochs, walltime, mem_gb in TIERS:
        for strat in strategies:
            label = f"r_d{d // 1000}k_{strat}" if d < 1_000_000 else f"r_d{d // 1_000_000}M_{strat}"
            txt, out_dir = slurm_text(d, strat, n_reps, ens, epochs, walltime, mem_gb, label)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            path = Path(f"/tmp/_{label}.sh")
            path.write_text(txt)
            if args.dry_run:
                print(
                    f"  [DRY] {label:<28} D={d:>8} ens={ens} reps={n_reps} ep={epochs} wt={walltime}"
                )
                continue
            r = subprocess.run([SBATCH, str(path)], capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                jid = r.stdout.strip().split()[-1]
                submitted += 1
                print(f"  OK  {label:<28} D={d:>8} ens={ens} reps={n_reps} -> {jid}")
            else:
                failed += 1
                print(f"  ERR {label}: {r.stderr.strip()[:200]}")
    print()
    print(f"Submitted: {submitted}  Failed: {failed}")


if __name__ == "__main__":
    main()
