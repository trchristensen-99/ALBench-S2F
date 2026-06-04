"""Submit LegNet chr-split scaling experiment: 2 labels × 7 N × 3 seeds = 42 jobs."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"

NS = [3197, 6395, 15987, 31974, 63949, 159871, 296382]  # last = max chr-split genomic train
SEEDS = [100, 200, 300]
LABEL_SOURCES = ["oracle", "real"]
EPOCHS = 15
PATIENCE = 5


def qos_for(N):
    if N <= 6395:
        return "fast", "04:00:00"
    if N <= 63949:
        return "default", "12:00:00"
    return "slow_nice", "48:00:00"


def make_sh(N, seed, label_source, qos, wt):
    label = f"lnchr_{label_source}_n{N}_s{seed}"
    out_dir = f"{REPO}/outputs/legnet_chrsplit_scaling/{label_source}/n{N}/seed{seed}"
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
#SBATCH --mem=80G
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
uv run --no-sync python experiments/train_chrsplit_legnet.py \\
  --N {N} --seed {seed} --label_source {label_source} \\
  --epochs {EPOCHS} --early_stop_patience {PATIENCE} \\
  --out_dir {out_dir}
""",
    )


def make_sh_gpu(N, seed, label_source, qos, wt, gpu_type):
    label = f"lnchr_{label_source}_n{N}_s{seed}"
    out_dir = f"{REPO}/outputs/legnet_chrsplit_scaling/{label_source}/n{N}/seed{seed}"
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
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
uv run --no-sync python experiments/train_chrsplit_legnet.py \\
  --N {N} --seed {seed} --label_source {label_source} \\
  --epochs {EPOCHS} --early_stop_patience {PATIENCE} \\
  --out_dir {out_dir}
""",
    )


def try_submit(N, seed, ls):
    label = f"lnchr_{ls}_n{N}_s{seed}"
    out_dir = Path(f"{REPO}/outputs/legnet_chrsplit_scaling/{ls}/n{N}/seed{seed}")
    if (out_dir / "model.npz").exists():
        return None, "exists"
    # qos chain depending on N
    primary_qos, primary_wt = qos_for(N)
    if primary_qos == "fast":
        chains = [("fast", "04:00:00"), ("default", "12:00:00"), ("slow_nice", "48:00:00")]
    elif primary_qos == "default":
        chains = [("default", "12:00:00"), ("slow_nice", "48:00:00")]
    else:
        chains = [("slow_nice", "48:00:00")]
    # Try H100 first across qos chain, then V100
    for gpu_type in ["h100", "v100"]:
        for qos, wt in chains:
            _, sh = make_sh_gpu(N, seed, ls, qos, wt, gpu_type)
            p = Path(f"/tmp/_lnchr_{label}.sh")
            p.write_text(sh)
            r = subprocess.run([SBATCH, str(p)], capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return r.stdout.strip().split()[-1], f"{qos}/{gpu_type}"
    return None, "FAIL"


def main():
    submitted = skipped = 0
    for ls in LABEL_SOURCES:
        for N in NS:
            for seed in SEEDS:
                jid, qos = try_submit(N, seed, ls)
                if jid is None and qos == "exists":
                    skipped += 1
                elif jid:
                    submitted += 1
                    print(f"  {ls} N={N} s={seed} -> {jid} [{qos}]")
                else:
                    print(f"  FAIL {ls} N={N} s={seed}")
    print(f"\nSubmitted: {submitted}  Skipped: {skipped}")


if __name__ == "__main__":
    main()
