"""Verification experiment: random reservoir at D=1M with multiple seeds.

Goal: check whether the EffBlock+muon OOD finding (Pearson ~0.80) is reproducible
or a flukish artifact of HP-search-seed=42 + reservoir-sampling-seed=42.

Design: 2 independent reps. Each rep uses a fresh (reservoir_seed, hp_seed):
  rep 1: seed=100
  rep 2: seed=200

For each rep, submit all 6 mixed6 strategies. Each job uses slow_nice qos
(D=1M → 48h walltime cap). Per-rep cache is generated first.

Outputs land in outputs/random_d1M_verify_e100/seed{S}/{strategy}/ — analyzable
the same way as the main full_sweep cells.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"
WALLTIME_SLOW_NICE = "48:00:00"
WALLTIME_FAST = "04:00:00"

SEEDS = [100, 200, 300]
D = 1_000_000
TASK = "k562"
ORACLE = "ag_s2"
ROUNDS = 2
PER_ROUND = 3
EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3

MIXED6 = [
    (
        "llm_autoresearch",
        {"LLM_PROMPT_STYLE": "default", "LLM_MODEL": "claude-opus-4-7", "LLM_USE_KB": "1"},
        "llm_default_opus",
    ),
    (
        "llm_autoresearch",
        {
            "LLM_PROMPT_STYLE": "diverse",
            "LLM_MODEL": "claude-sonnet-4-5-20250929",
            "LLM_USE_KB": "1",
        },
        "llm_diverse_sonnet",
    ),
    (
        "llm_autoresearch",
        {
            "LLM_PROMPT_STYLE": "exploit",
            "LLM_MODEL": "claude-sonnet-4-5-20250929",
            "LLM_USE_KB": "1",
        },
        "llm_exploit_sonnet",
    ),
    ("autoresearch_batch", {}, "ar_batch"),
    ("autoresearch_massive", {}, "ar_massive"),
    ("random", {}, "random"),
]


def cache_gen_sh(seed: int, cache_path: str) -> tuple[str, str]:
    label = f"genv_random_d{D}_seed{seed}"
    return (
        label,
        f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time={WALLTIME_FAST}
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
  --task {TASK} --reservoir random --D {D} --seed {seed} \\
  --oracle {ORACLE} --out {cache_path}
""",
    )


def search_sh(
    seed: int, strat: str, env: dict, sub: str, cache_path: str, dep_jobid: str | None
) -> tuple[str, str]:
    label = f"randv_d{D}_seed{seed}_{sub}"
    out_dir = f"{REPO}/outputs/random_d1M_verify_e100/seed{seed}/{sub}"
    env_lines = "\n".join(f'export {k}="{v}"' for k, v in env.items())
    dep_line = f"#SBATCH --dependency=afterok:{dep_jobid}" if dep_jobid else ""
    return (
        label,
        f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time={WALLTIME_SLOW_NICE}
#SBATCH --mem=80G
{dep_line}
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD"
export TQDM_DISABLE=1
{env_lines}
uv run --no-sync python experiments/scaling_hp_search.py \\
  --strategies {strat} --rounds {ROUNDS} --per_strategy_per_round {PER_ROUND} \\
  --D {D} --ref_only --epochs {EPOCHS} \\
  --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA} \\
  --hp_seed {seed} \\
  --reservoir_cache {cache_path} \\
  --out_dir {out_dir}
""",
    )


def sbatch_one(label: str, script: str) -> str | None:
    p = Path(f"/tmp/_rv_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split()[-1]
    print(f"  ERR {label}: {r.stderr.strip()[:150]}")
    return None


def main():
    for seed in SEEDS:
        cache_path = f"{REPO}/outputs/reservoir_cache/k562_random_d{D}_seed{seed}.npz"
        if not Path(cache_path).exists():
            lbl, sh = cache_gen_sh(seed, cache_path)
            jid = sbatch_one(lbl, sh)
            print(f"  cache_gen seed={seed} -> {jid}")
            dep = jid
        else:
            print(f"  cache exists seed={seed}")
            dep = None
        for strat, env, sub in MIXED6:
            lbl, sh = search_sh(seed, strat, env, sub, cache_path, dep)
            jid = sbatch_one(lbl, sh)
            print(f"    {sub} seed={seed} -> {jid}")


if __name__ == "__main__":
    main()
