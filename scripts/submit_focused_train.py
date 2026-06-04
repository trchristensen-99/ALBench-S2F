"""Submit focused training: 3 fixed HP configs × 2 reservoir seeds across
6 reservoirs × 7 D values. Cache_gens for seed=100 fire first, then training
jobs wait via afterok dependency.

Total: ~35 cache_gens + ~246 training jobs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"

DS = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
RESERVOIRS = os.environ.get(
    "FT_RESERVOIRS", "genomic,random,prm_1pct,prm_10pct,evoaug_heavy,motif_shuffled"
).split(",")
GENOMIC_MAX_D = 300_000
CONFIGS = os.environ.get("FT_CONFIGS", "A,C,D").split(",")
SEEDS = [int(s) for s in os.environ.get("FT_SEEDS", "42,100").split(",")]
NICE = int(os.environ.get("FT_NICE", "0"))
TASK = "k562"
ORACLE = "ag_s2"
EPOCHS = 15
PATIENCE = 5


def walltime(d: int) -> tuple[str, str]:
    """(qos, walltime) — max walltime per qos."""
    if d <= 3_000:
        return "fast", "04:00:00"
    if d <= 100_000:
        return "default", "12:00:00"
    return "slow_nice", "48:00:00"


def qos_chain(d: int) -> list[tuple[str, str]]:
    """Ordered list of (qos, walltime) to try; later qos = overflow."""
    if d <= 3_000:
        return [("fast", "04:00:00"), ("default", "12:00:00"), ("slow_nice", "48:00:00")]
    if d <= 100_000:
        return [("default", "12:00:00"), ("slow_nice", "48:00:00")]
    return [("slow_nice", "48:00:00")]


def cache_gen_sh(reservoir: str, D: int, seed: int, cache_path: str) -> tuple[str, str]:
    qos, wt = walltime(D)
    label = f"ft_gen_{reservoir}_d{D}_seed{seed}"
    return (
        label,
        f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos={qos}
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time={wt}
#SBATCH --mem=100G
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD"
uv run --no-sync python scripts/generate_reservoir_cache.py \\
  --task {TASK} --reservoir {reservoir} --D {D} --seed {seed} \\
  --oracle {ORACLE} --out {cache_path}
""",
    )


def train_sh(
    reservoir: str,
    D: int,
    seed: int,
    config_id: str,
    cache_path: str | None,
    dep_jobid: str | None,
    qos: str,
    wt: str,
    gpu_type: str = "h100",
) -> tuple[str, str]:
    label = f"ft_{reservoir}_d{D}_seed{seed}_{config_id}"
    out_dir = f"{REPO}/outputs/focused_train/k562_{reservoir}_d{D}_seed{seed}/config_{config_id}"
    dep_line = f"#SBATCH --dependency=afterok:{dep_jobid}" if dep_jobid else ""
    cache_arg = f"--reservoir_cache {cache_path}" if cache_path else ""
    chr_val_arg = "--chr_val" if reservoir != "random" else ""
    return (
        label,
        f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{gpu_type}:1
#SBATCH --cpus-per-task=8
#SBATCH --time={wt}
#SBATCH --mem=80G
#SBATCH --nice={NICE}
{dep_line}
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
uv run --no-sync python experiments/train_one_config.py \\
  --reservoir {reservoir} --D {D} --config_id {config_id} \\
  --seed {seed} {cache_arg} {chr_val_arg} \\
  --epochs {EPOCHS} --early_stop_patience {PATIENCE} \\
  --out_dir {out_dir}
""",
    )


def try_submit_train(reservoir, D, seed, cfg, cache_path, dep):
    """Try submission across qos chain (H100, then V100)."""
    for gpu in ["h100", "v100"]:
        for qos, wt in qos_chain(D):
            label, sh = train_sh(reservoir, D, seed, cfg, cache_path, dep, qos, wt, gpu)
            jid = sbatch(label, sh)
            if jid:
                return jid, qos, gpu
    return None, None, None


def sbatch(label: str, script: str) -> str | None:
    p = Path(f"/tmp/_ft_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split()[-1]
    print(f"  ERR {label}: {r.stderr.strip()[:150]}")
    return None


def main():
    # Discover existing job names + summaries to avoid dup submissions
    r = subprocess.run(
        [
            "/cm/shared/apps/slurm/current/bin/squeue",
            "--me",
            "-h",
            "--format=%j",
            "--states=PD,R,CF",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    inflight = set(line.strip() for line in r.stdout.strip().split("\n") if line.strip())

    cache_jobids: dict[tuple[str, int, int], str] = {}  # (reservoir, D, seed) -> job_id
    n_cache_submitted = 0
    n_train_submitted = 0
    n_skipped = 0

    # 1) Cache generation for seed=100 (non-genomic only; seed=42 reuses existing caches)
    for r in RESERVOIRS:
        if r == "genomic":
            continue
        for D in DS:
            for seed in SEEDS:
                cache_path = f"{REPO}/outputs/reservoir_cache/k562_{r}_d{D}_seed{seed}.npz"
                if Path(cache_path).exists():
                    cache_jobids[(r, D, seed)] = None  # cache exists, no dep
                    continue
                lbl, sh = cache_gen_sh(r, D, seed, cache_path)
                if lbl in inflight:
                    n_skipped += 1
                    cache_jobids[(r, D, seed)] = None
                    continue
                jid = sbatch(lbl, sh)
                if jid:
                    cache_jobids[(r, D, seed)] = jid
                    n_cache_submitted += 1

    # 2) Training jobs
    for r_ in RESERVOIRS:
        for D in DS:
            if r_ == "genomic" and D > GENOMIC_MAX_D:
                continue
            for seed in SEEDS:
                cache_path = None
                dep = None
                if r_ != "genomic":
                    cache_path = f"{REPO}/outputs/reservoir_cache/k562_{r_}_d{D}_seed{seed}.npz"
                    dep = cache_jobids.get((r_, D, seed))
                for cfg in CONFIGS:
                    lbl = f"ft_{r_}_d{D}_seed{seed}_{cfg}"
                    out_dir = Path(
                        f"{REPO}/outputs/focused_train/k562_{r_}_d{D}_seed{seed}/config_{cfg}"
                    )
                    if (
                        lbl in inflight
                        or (out_dir / "model.npz").exists()
                        or (out_dir / "skipped.txt").exists()
                    ):
                        n_skipped += 1
                        continue
                    jid, qos_used, gpu_used = try_submit_train(r_, D, seed, cfg, cache_path, dep)
                    if jid:
                        n_train_submitted += 1
                        print(f"  {lbl} -> {jid} [{qos_used}/{gpu_used}]")

    print(f"\nCache_gens submitted: {n_cache_submitted}")
    print(f"Training jobs submitted: {n_train_submitted}")
    print(f"Skipped (in-flight or done): {n_skipped}")


if __name__ == "__main__":
    main()
