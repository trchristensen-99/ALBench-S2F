"""LLM PROMPT ABLATION — a focused follow-up to the Phase-0 prompt screen that
isolates the two factors that confound the LLM family's bake-off comparison:

  (A) search GUIDANCE   — how (or whether) we tell the model to proceed, and
  (B) background CONTEXT — how much domain steering it sees (ADVANCED_GUIDANCE,
                           cross-experiment KB priors, editorial closers).

WHY scoped, not a full cross: per the screen's partial standings the well-sampled
winners are the `default` and `exploit` styles, so we vary CONTEXT on THOSE only
(full → nokb → none) and add two controls — `blank` (black-box: no domain framing
at all) and `misguided` (deliberately wrong guidance). Held to ONE model + ONE seed
+ genomic D=30k at SHALLOW depth (oracle-Pearson plateaus by round ~8-10), so the
whole ablation is a handful of cells and cheap on the Claude cap.

Interpretation: if `default/full` (rich) beats `blank` (black-box), the SCAFFOLDING
is doing the work — the LLM's bake-off edge over evo/random is partly human-injected
domain knowledge + prior-run KB those strategies don't get (a fairness confound). If
`blank` matches, the edge is intrinsic in-context optimization and the comparison is
clean. `misguided` vs `blank` tests whether the model anchors on bad guidance or
overrides it from its own observations.

CELLS (style, context):
  (default, full)  (default, nokb)  (default, none)
  (exploit, full)  (exploit, nokb)  (exploit, none)
  (blank, none)    (misguided, none)        # no-prior styles force context=none

COST: every cell calls Claude (CONSUMES the usage cap). n_calls ≈ n_cells × ROUNDS.
Print the estimate; DRY_RUN=1 to plan only; SMOKE_ONLY=1 for 1 cell × 2 rounds.

Usage:
  DRY_RUN=1   python scripts/submit_llm_prompt_ablation.py
  SMOKE_ONLY=1 python scripts/submit_llm_prompt_ablation.py
  python scripts/submit_llm_prompt_ablation.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

OUT_ROOT = os.environ.get("LLM_ABL_OUT_ROOT", f"{REPO}/outputs/hp_llm_ablation_e100")

# Curated (style, context). No-prior styles (blank/misguided/neutral) force "none".
CELLS: list[tuple[str, str]] = [
    ("default", "full"),
    ("default", "nokb"),
    ("default", "none"),
    ("exploit", "full"),
    ("exploit", "nokb"),
    ("exploit", "none"),
    ("blank", "none"),
    ("misguided", "none"),
]

MODEL = os.environ.get("LLM_ABL_MODEL", "claude-sonnet-4-6")
NOVEL = os.environ.get("LLM_ABL_NOVEL", "0")  # hold off so CONTEXT alone controls background
RESERVOIR = "genomic"
D = int(os.environ.get("LLM_ABL_D", "30000"))
DATA_SEED, HP_SEED = 42, 0
ROUNDS = int(os.environ.get("LLM_ABL_ROUNDS", "10"))  # plateau hits by ~r8-10
PER_ROUND = int(os.environ.get("LLM_ABL_PER_ROUND", "2"))
TEMPERATURE = os.environ.get("LLM_ABL_TEMPERATURE", "")

EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3

SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def variant_label(style: str, context: str) -> str:
    return f"llm_{style}_ctx{context}"


def out_dir(style: str, context: str) -> Path:
    return Path(
        f"{OUT_ROOT}/k562_{RESERVOIR}_d{D}/seed{DATA_SEED}_{HP_SEED}/{variant_label(style, context)}"
    )


def is_complete(od: Path, rounds: int) -> bool:
    if not od.exists():
        return False
    if (od / ".ablation_done").exists():
        return True
    return len(list(od.glob("*_meta.json"))) >= rounds * PER_ROUND


def job_script(style: str, context: str, rounds: int, qos: str, wt: str) -> tuple[str, str]:
    var = variant_label(style, context)
    label = f"lpa_{RESERVOIR}_d{D}_{var}_s{DATA_SEED}_{HP_SEED}"
    od = out_dir(style, context)
    temp_env = f'export LLM_TEMPERATURE="{TEMPERATURE}"\n' if TEMPERATURE else ""
    script = f"""#!/bin/bash
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
#SBATCH --requeue
set -uo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
echo "=== job start (SLURM_RESTART_COUNT=${{SLURM_RESTART_COUNT:-0}}; resume preloads on-disk history) ==="
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export HP_FAST=1
export HP_CACHE_DIR="$PWD/outputs/tensor_cache"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export LLM_MODEL="{MODEL}"
export LLM_PROMPT_STYLE="{style}"
export LLM_CONTEXT="{context}"
export LLM_ALLOW_NOVEL_AXES="{NOVEL}"
{temp_env}ATTEMPT=0
while true; do
  uv run --no-sync python experiments/scaling_hp_search.py \\
    --strategies llm_autoresearch --rounds {rounds} --per_strategy_per_round {PER_ROUND} \\
    --D {D} --ref_only --chr_val \\
    --data_seed {DATA_SEED} --hp_seed {HP_SEED} \\
    --epochs {EPOCHS} --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA} \\
    --out_dir {od}
  rc=$?
  if [ $rc -eq 0 ]; then touch {od}/.ablation_done; echo "=== DONE rc=0 ==="; break; fi
  ATTEMPT=$((ATTEMPT+1))
  if [ $rc -eq 42 ]; then
    if [ $ATTEMPT -ge 12 ]; then echo "=== too many rate-limit stops; giving up ==="; exit 42; fi
    echo "=== rate-limit budget exhausted (exit 42); sleep 1800s then resume (attempt $ATTEMPT) ==="
    sleep 1800; continue
  fi
  echo "=== FAILED rc=$rc ==="; exit $rc
done
"""
    return label, script


def sbatch(label: str, script: str) -> tuple[str | None, str]:
    p = Path(f"/tmp/_lpa_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main() -> None:
    smoke = os.environ.get("SMOKE_ONLY") == "1"
    dry = os.environ.get("DRY_RUN") == "1"
    cells = CELLS[:1] if smoke else CELLS
    rounds = 2 if smoke else ROUNDS

    est_calls = len(cells) * rounds
    print(f"=== LLM prompt ablation: {len(cells)} cells × {rounds} rounds ===")
    print(
        f"=== model={MODEL} novel={NOVEL} reservoir={RESERVOIR} D={D} seed={DATA_SEED}:{HP_SEED} ==="
    )
    print(f"=== ~{est_calls} Claude CLI calls (CONSUMES the usage cap) ===")
    for style, context in cells:
        print(f"  {variant_label(style, context):28s} style={style} context={context}")
    if dry:
        print("=== DRY_RUN=1: nothing submitted ===")
        return

    r = subprocess.run(
        [SQUEUE, "--me", "-h", "--format=%j|%q", "--states=PD,R,CF"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    inflight = set()
    qcount = {"fast": 0, "default": 0, "slow_nice": 0}
    for ln in r.stdout.strip().split("\n"):
        if not ln.strip():
            continue
        name, _, q = ln.partition("|")
        inflight.add(name.strip())
        if q.strip() in qcount:
            qcount[q.strip()] += 1

    n_sub = n_skip = n_done = 0
    for style, context in cells:
        var = variant_label(style, context)
        label = f"lpa_{RESERVOIR}_d{D}_{var}_s{DATA_SEED}_{HP_SEED}"
        od = out_dir(style, context)
        if is_complete(od, rounds):
            n_done += 1
            continue
        if label in inflight:
            n_skip += 1
            continue
        submitted = False
        # Small D → default (12h) first, then slow_nice. LLM never uses fast.
        for qos, wt in [("default", "12:00:00"), ("slow_nice", "2-00:00:00")]:
            if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                continue
            lbl, sh = job_script(style, context, rounds, qos, wt)
            jid, err = sbatch(lbl, sh)
            if jid:
                n_sub += 1
                qcount[qos] = qcount.get(qos, 0) + 1
                print(f"  SUB {label} -> {jid} [{qos}]")
                submitted = True
                break
            print(f"  ERR {label} [{qos}]: {err[:160]}")
        if not submitted:
            print(f"  HOLD {label}: all qos at cap")
    print(f"=== submitted={n_sub} skip_inflight={n_skip} already_done={n_done} ===")


if __name__ == "__main__":
    main()
