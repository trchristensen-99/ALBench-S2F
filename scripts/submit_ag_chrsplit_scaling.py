"""Submit AG S1 chr-split scaling: 2 labels × 7 N × 3 seeds = 42 jobs."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"

NS = [3197, 6395, 15987, 31974, 63949, 159871, 296382]
SEEDS = [100, 200, 300]
LABEL_SOURCES = ["oracle", "real"]


def qos_for(N):
    # Fill all qos tiers to use fast/default capacity
    if N <= 6395:
        return "fast", "04:00:00"
    if N <= 63949:
        return "default", "12:00:00"
    return "slow_nice", "48:00:00"


def make_sh(N, seed, label_source, qos, wt):
    label = f"agchr_{label_source}_n{N}_s{seed}"
    out_dir = f"{REPO}/outputs/ag_chrsplit_scaling/{label_source}/n{N}/seed{seed}"
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
#SBATCH --mem=120G
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD"
uv run --no-sync python experiments/train_chrsplit_ag_s1.py \\
  --N {N} --seed {seed} --label_source {label_source} \\
  --out_dir {out_dir}
""",
    )


def make_sh_gpu(N, seed, label_source, qos, wt, gpu_type):
    label = f"agchr_{label_source}_n{N}_s{seed}"
    out_dir = f"{REPO}/outputs/ag_chrsplit_scaling/{label_source}/n{N}/seed{seed}"
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
#SBATCH --mem=120G
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD"
uv run --no-sync python experiments/train_chrsplit_ag_s1.py \\
  --N {N} --seed {seed} --label_source {label_source} \\
  --out_dir {out_dir}
""",
    )


def qos_chain(N):
    if N <= 6395:
        return [("fast", "04:00:00"), ("default", "12:00:00"), ("slow_nice", "48:00:00")]
    if N <= 63949:
        return [("default", "12:00:00"), ("slow_nice", "48:00:00")]
    return [("slow_nice", "48:00:00")]


def get_inflight_jobs():
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
    return set(line.strip() for line in r.stdout.strip().split("\n") if line.strip())


def main():
    inflight = get_inflight_jobs()
    submitted = skipped = 0
    for ls in LABEL_SOURCES:
        for N in NS:
            for seed in SEEDS:
                label = f"agchr_{ls}_n{N}_s{seed}"
                out_dir = Path(f"{REPO}/outputs/ag_chrsplit_scaling/{ls}/n{N}/seed{seed}")
                if (out_dir / "summary.json").exists() or label in inflight:
                    skipped += 1
                    continue
                jid = None
                for gpu_type in ["h100", "v100"]:
                    for qos, wt in qos_chain(N):
                        _, sh = make_sh_gpu(N, seed, ls, qos, wt, gpu_type)
                        p = Path(f"/tmp/_agchr_{ls}_n{N}_s{seed}.sh")
                        p.write_text(sh)
                        r = subprocess.run(
                            [SBATCH, str(p)], capture_output=True, text=True, timeout=30
                        )
                        if r.returncode == 0:
                            jid = r.stdout.strip().split()[-1]
                            print(f"  {ls} N={N} s={seed} -> {jid} [{qos}/{gpu_type}]")
                            submitted += 1
                            break
                    if jid:
                        break
                if jid is None:
                    print(f"  FAIL {ls} N={N} s={seed}")
    print(f"\nSubmitted: {submitted}  Skipped (done or in-flight): {skipped}")


if __name__ == "__main__":
    main()
