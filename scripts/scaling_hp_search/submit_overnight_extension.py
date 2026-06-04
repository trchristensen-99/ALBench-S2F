"""Overnight extension — fill remaining D values + start reservoir sampling.

Submits HP search at D=3k, D=10k, D=1M (the missing endpoints) + a genomic
reservoir probe at D=30k using known-good HP configs.

Storage: ~30-50MB per HP-search job, ~200MB per reservoir job (with weights).
Should consume <1GB total, comfortable in current 278GB free.

Usage:
    python scripts/scaling_hp_search/submit_overnight_extension.py [--dry_run]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
SBATCH = "/cm/shared/apps/slurm/current/bin/sbatch"

# (D, strategy, rounds, per_round, epochs, qos, walltime, env, label)
JOB_MATRIX = [
    # D=3k — low-end cheap probe (mixed4 reduced)
    (
        3_000,
        "llm_autoresearch",
        3,
        3,
        30,
        "fast",
        "04:00:00",
        {"LLM_PROMPT_STYLE": "default", "LLM_MODEL": "claude-opus-4-7", "LLM_USE_KB": "1"},
        "p4d_d3k_llm_default_opus",
    ),
    (3_000, "autoresearch_massive", 3, 3, 30, "fast", "04:00:00", {}, "p4d_d3k_ar_massive"),
    (3_000, "random", 3, 3, 30, "default", "04:00:00", {}, "p4d_d3k_random"),
    # D=10k — mixed4 reduced
    (
        10_000,
        "llm_autoresearch",
        5,
        3,
        30,
        "default",
        "12:00:00",
        {"LLM_PROMPT_STYLE": "default", "LLM_MODEL": "claude-opus-4-7", "LLM_USE_KB": "1"},
        "p4d_d10k_llm_default_opus",
    ),
    (
        10_000,
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
        "p4d_d10k_llm_diverse_sonnet",
    ),
    (10_000, "autoresearch_massive", 5, 3, 30, "slow_nice", "12:00:00", {}, "p4d_d10k_ar_massive"),
    (10_000, "random", 5, 3, 30, "slow_nice", "12:00:00", {}, "p4d_d10k_random"),
    # D=1M — critical scaling endpoint, highly reduced (only top-2 strategies, 15 epochs)
    # 2 rounds × 3 = 6 models; ~2hr/model × 6 = 12hr per job → walltime 48h for safety
    (
        1_000_000,
        "llm_autoresearch",
        2,
        3,
        15,
        "slow_nice",
        "48:00:00",
        {"LLM_PROMPT_STYLE": "default", "LLM_MODEL": "claude-opus-4-7", "LLM_USE_KB": "1"},
        "p4d_d1M_llm_default_opus",
    ),
    (
        1_000_000,
        "autoresearch_massive",
        2,
        3,
        15,
        "slow_nice",
        "48:00:00",
        {},
        "p4d_d1M_ar_massive",
    ),
]


def slurm_text(d, strategy, rounds, per_round, epochs, qos, walltime, env_overrides, label):
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
  echo "ERROR: no LLM token. Aborting."
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
            print(f"  [DRY] {label:<32} D={d:>8} {strat:<22} {rounds}r×{per_round}c qos={qos:<10}")
            continue
        r = subprocess.run([SBATCH, str(path)], capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            jid = r.stdout.strip().split()[-1]
            submitted.append((label, jid))
            print(f"  ✓ {label:<32} D={d:>8} {strat:<22} qos={qos:<10} → job {jid}")
        else:
            failures.append((label, r.stderr.strip()[:200]))
            print(f"  ✗ {label}: {r.stderr.strip()[:200]}")
    print()
    print(f"Submitted: {len(submitted)} | Failed: {len(failures)}")
    if failures:
        for label, err in failures:
            print(f"  - {label}: {err}")


if __name__ == "__main__":
    main()
