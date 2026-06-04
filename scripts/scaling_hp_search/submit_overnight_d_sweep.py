"""Overnight job matrix — answer 'how much HP search is still needed at large D?'

Submits Phase 4 (D=30k, full mixed6 with expanded HPs) + cross-D search at D=100k
and D=300k with progressively reduced budgets. Tomorrow's analysis compares
ranking stability across D to decide the long-D pipeline.

Usage:
    python scripts/scaling_hp_search/submit_overnight_d_sweep.py [--dry_run]
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"

# (D, strategy_list, rounds, per_round, epochs, qos, walltime_hours)
JOB_MATRIX = [
    # Phase 4 — D=30k mixed6 with expanded HP space (block_class ∈ {eff, ag, plain}, optimizer ∈ {adam, adamw, muon})
    # 6 strategies × 10 rounds × 3 = 180 models
    (
        30_000,
        "llm_autoresearch",
        10,
        3,
        30,
        "fast",
        "04:00:00",
        {"LLM_PROMPT_STYLE": "default", "LLM_MODEL": "claude-opus-4-7", "LLM_USE_KB": "1"},
        "p4_d30k_llm_default_opus",
    ),
    (
        30_000,
        "llm_autoresearch",
        10,
        3,
        30,
        "fast",
        "04:00:00",
        {
            "LLM_PROMPT_STYLE": "diverse",
            "LLM_MODEL": "claude-sonnet-4-5-20250929",
            "LLM_USE_KB": "1",
        },
        "p4_d30k_llm_diverse_sonnet",
    ),
    (
        30_000,
        "llm_autoresearch",
        10,
        3,
        30,
        "default",
        "12:00:00",
        {
            "LLM_PROMPT_STYLE": "exploit",
            "LLM_MODEL": "claude-sonnet-4-5-20250929",
            "LLM_USE_KB": "1",
        },
        "p4_d30k_llm_exploit_sonnet",
    ),
    (30_000, "autoresearch_massive", 10, 3, 30, "default", "12:00:00", {}, "p4_d30k_ar_massive"),
    (30_000, "autoresearch_batch", 10, 3, 30, "default", "12:00:00", {}, "p4_d30k_ar_batch"),
    (30_000, "random", 10, 3, 30, "slow_nice", "12:00:00", {}, "p4_d30k_random"),
    # Cross-D — D=100k mixed6 reduced budget (5 rounds × 3 = 15 models per strategy = 90 total)
    (
        100_000,
        "llm_autoresearch",
        5,
        3,
        30,
        "slow_nice",
        "12:00:00",
        {"LLM_PROMPT_STYLE": "default", "LLM_MODEL": "claude-opus-4-7", "LLM_USE_KB": "1"},
        "p4d_d100k_llm_default_opus",
    ),
    (
        100_000,
        "llm_autoresearch",
        5,
        3,
        30,
        "slow_nice",
        "12:00:00",
        {
            "LLM_PROMPT_STYLE": "diverse",
            "LLM_MODEL": "claude-sonnet-4-5-20250929",
            "LLM_USE_KB": "1",
        },
        "p4d_d100k_llm_diverse_sonnet",
    ),
    (
        100_000,
        "llm_autoresearch",
        5,
        3,
        30,
        "slow_nice",
        "12:00:00",
        {
            "LLM_PROMPT_STYLE": "exploit",
            "LLM_MODEL": "claude-sonnet-4-5-20250929",
            "LLM_USE_KB": "1",
        },
        "p4d_d100k_llm_exploit_sonnet",
    ),
    (
        100_000,
        "autoresearch_massive",
        5,
        3,
        30,
        "slow_nice",
        "12:00:00",
        {},
        "p4d_d100k_ar_massive",
    ),
    (100_000, "autoresearch_batch", 5, 3, 30, "slow_nice", "12:00:00", {}, "p4d_d100k_ar_batch"),
    (100_000, "random", 5, 3, 30, "slow_nice", "12:00:00", {}, "p4d_d100k_random"),
    # Cross-D — D=300k highly reduced (3 rounds × 3 = 9 models per strategy = 27 total)
    # Focused on top 3 strategies from Phase 2 (opus, ar_massive, random)
    (
        300_000,
        "llm_autoresearch",
        3,
        3,
        25,
        "slow_nice",
        "12:00:00",
        {"LLM_PROMPT_STYLE": "default", "LLM_MODEL": "claude-opus-4-7", "LLM_USE_KB": "1"},
        "p4d_d300k_llm_default_opus",
    ),
    (
        300_000,
        "autoresearch_massive",
        3,
        3,
        25,
        "slow_nice",
        "12:00:00",
        {},
        "p4d_d300k_ar_massive",
    ),
    (300_000, "random", 3, 3, 25, "slow_nice", "12:00:00", {}, "p4d_d300k_random"),
]


def slurm_text(
    d: int,
    strategy: str,
    rounds: int,
    per_round: int,
    epochs: int,
    qos: str,
    walltime: str,
    env_overrides: dict,
    label: str,
) -> tuple[str, str]:
    out_dir = f"{REPO}/outputs/phase4_overnight/{label}"
    env_lines = "\n".join(f'export {k}="{v}"' for k, v in env_overrides.items())
    return (
        f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos={qos}
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time={walltime}
#SBATCH --mem=80G
#SBATCH --export=ALL
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD"
{env_lines}
if [ "{strategy}" = "llm_autoresearch" ] && [ -z "${{CLAUDE_CODE_OAUTH_TOKEN:-}}" ] && [ -z "${{ANTHROPIC_API_KEY:-}}" ]; then
  echo "ERROR: neither CLAUDE_CODE_OAUTH_TOKEN nor ANTHROPIC_API_KEY set. Aborting."
  exit 1
fi
uv run --no-sync python experiments/scaling_hp_search.py \\
  --strategies {strategy} \\
  --rounds {rounds} \\
  --per_strategy_per_round {per_round} \\
  --D {d} \\
  --ref_only \\
  --epochs {epochs} \\
  --out_dir {out_dir}
""",
        out_dir,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    submitted = []
    failures = []
    for d, strat, rounds, per_round, epochs, qos, walltime, env, label in JOB_MATRIX:
        txt, out_dir = slurm_text(d, strat, rounds, per_round, epochs, qos, walltime, env, label)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(f"/tmp/_{label}.sh")
        path.write_text(txt)
        if args.dry_run:
            print(f"  [DRY] {label:<34} D={d:>7} {strat:<22} {rounds}r×{per_round}c qos={qos:<10}")
            continue
        r = subprocess.run([SBATCH, str(path)], capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            jid = r.stdout.strip().split()[-1]
            submitted.append((label, jid))
            print(f"  ✓ {label:<34} D={d:>7} {strat:<22} qos={qos:<10} → job {jid}")
        else:
            failures.append((label, r.stderr.strip()[:200]))
            print(f"  ✗ {label}: {r.stderr.strip()[:200]}")
    print()
    print(f"Submitted: {len(submitted)}")
    print(f"Failed:    {len(failures)}")
    if failures:
        for label, err in failures:
            print(f"  - {label}: {err}")


if __name__ == "__main__":
    main()
