"""Launch the ROUNDS-SCALING study: run each HP strategy for many more rounds on a
few representative (reservoir × D) cells, to measure how best val-Pearson improves
with search rounds and pick the optimal number of rounds.

Mirrors scripts/submit_hp_search.py job construction exactly (same scaling_hp_search
invocation, qos chain, MaxSubmitPU caps, exit-42 rate-limit loop, resume-via-meta),
but:
  • writes to a SEPARATE namespace  outputs/hp_rounds_scaling/  (never collides with
    the main 240-cell run),
  • names jobs  rsc_*  (so the main hp_watchdog / hp_autodeploy ignore them, and this
    study's own controller can count only its jobs),
  • fixes the study grid: 2 reservoirs × {30k, 300k} × 4 strategy variants,
  • runs ROUNDS rounds (env HP_ROUNDS, default 50) instead of the main run's 4.

Idempotent + resumable (skips complete + in-flight cells), so the rounds-scaling
watchdog can re-run it every interval to self-heal.

Env overrides: HP_ROUNDS, HP_PER_ROUND, RSC_RESERVOIRS, RSC_DS.

Run:  python scripts/submit_rounds_scaling.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
BIN = "/cm/shared/apps/slurm/current/bin"
SBATCH = f"{BIN}/sbatch"
SQUEUE = f"{BIN}/squeue"

# Study grid (deliberately small + contrasting): a neutral genomic pool vs a
# synthetic motif-planted pool, at one mid and one large D.
RESERVOIRS = os.environ.get("RSC_RESERVOIRS", "mixed_genomic_random,motif_planted_v2").split(",")
DS = [int(x) for x in os.environ.get("RSC_DS", "30000,300000").split(",")]

# GPU type: h100 (default, preferred — faster) or v100 (idle backfill while the
# H100-only main search runs; ~2-3x slower, fine at D=30k). V100 jobs are forced to
# slow_nice and skip the H100-only node exclude. See feedback-h100-preference.
GPU = os.environ.get("RSC_GPU", "h100").lower()

# Long horizon: ~50 rounds × 2/round = 100 LLM models per cell (vs 8 in the main
# run) — enough to see the plateau. Tunable before the controller fires.
ROUNDS = int(os.environ.get("HP_ROUNDS", "50"))
PER_ROUND = int(os.environ.get("HP_PER_ROUND", "2"))
EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-3
DATA_SEED = 42
HP_SEED = 0
POOL_D = 1_000_000

# (variant_label, strategies, llm_model, llm_style). llm_model/style empty for algo.
LLM_VARIANTS = [
    ("llm_default", "llm_autoresearch", "claude-opus-4-7", "default"),
    ("llm_explore", "llm_autoresearch", "claude-sonnet-4-6", "explore"),
    ("llm_exploit", "llm_autoresearch", "claude-sonnet-4-6", "exploit"),
]
ALGO_VARIANT = ("algo", "autoresearch_batch,autoresearch_massive,random", "", "")
ALL_VARIANTS = LLM_VARIANTS + [ALGO_VARIANT]

# Optional variant filter (e.g. RSC_VARIANTS=algo to run only the non-LLM variant during
# V100 backfill, so we don't contend on the Claude API quota with the running main search).
_sel = os.environ.get("RSC_VARIANTS", "").strip()
if _sel:
    _keep = set(_sel.split(","))
    ALL_VARIANTS = [v for v in ALL_VARIANTS if v[0] in _keep]


def pool_cache(reservoir: str) -> str:
    return f"{REPO}/outputs/reservoir_cache/k562_{reservoir}_d{POOL_D}_seed{DATA_SEED}.npz"


def n_strategies(strategies: str) -> int:
    return len(strategies.split(","))


def expected_models(strategies: str) -> int:
    return n_strategies(strategies) * ROUNDS * PER_ROUND


# Fresh root for the epochs=100 regime (no-mixing guard); override via HP_OUT_ROOT.
OUT_ROOT = os.environ.get("HP_OUT_ROOT", f"{REPO}/outputs/hp_rounds_scaling_e100")


def val_protocol() -> str:
    """Validation methodology (see submit_hp_search.val_protocol).

    'holdout' (default, 2026-06-03) = per-combo 10% random holdout, no --chr_val.
    'chr_val' (legacy) = genomic chr19/21/X holdout. The in-flight rounds-scaling
    study is PINNED to its original protocol via `<OUT_ROOT>/.val_protocol` so a
    watchdog resubmit never mixes val methodology within the study.
    Env HP_VAL_PROTOCOL overrides.
    """
    env = os.environ.get("HP_VAL_PROTOCOL", "").strip()
    if env:
        return env
    marker = Path(OUT_ROOT) / ".val_protocol"
    if marker.exists():
        tok = marker.read_text().strip().split()
        if tok:
            return tok[0]
    return "holdout"


def out_dir(reservoir: str, D: int, variant: str) -> Path:
    return Path(f"{OUT_ROOT}/k562_{reservoir}_d{D}/{variant}")


def is_complete(od: Path, strategies: str) -> bool:
    if not od.exists():
        return False
    return len(list(od.glob("*_meta.json"))) >= expected_models(strategies)


def qos_walltime(D: int, is_llm: bool) -> tuple[str, str]:
    if is_llm:
        if D <= 100_000:
            return "default", "12:00:00"
        return "slow_nice", "2-00:00:00"
    if D <= 10_000:
        return "fast", "04:00:00"
    if D <= 100_000:
        return "default", "12:00:00"
    return "slow_nice", "2-00:00:00"


def qos_chain(D: int, is_llm: bool) -> list[tuple[str, str]]:
    pref_qos, pref_wt = qos_walltime(D, is_llm)
    chain = [(pref_qos, pref_wt)]
    if pref_qos != "slow_nice":
        chain.append(("slow_nice", "2-00:00:00"))
    return chain


SUBMIT_CAP = {"fast": 16, "default": 16, "slow_nice": 1000}


def job_script(reservoir, D, variant, strategies, model, style, cache_path, qos, wt):
    is_llm = bool(model)
    label = f"rsc_{reservoir}_d{D}_{variant}"
    od = out_dir(reservoir, D, variant)
    chr_val_arg = "--chr_val" if (val_protocol() == "chr_val" and reservoir != "random") else ""
    # bamgpu15/25 are flaky H100 nodes excluded from the main run; the exclude is
    # only meaningful for H100 jobs.
    exclude_line = "#SBATCH --exclude=bamgpu15,bamgpu25" if GPU == "h100" else ""
    llm_env = ""
    if is_llm:
        llm_env = f'export LLM_MODEL="{model}"\nexport LLM_PROMPT_STYLE="{style}"'
    script = f"""#!/bin/bash
#SBATCH --job-name={label}
#SBATCH --output={REPO}/logs/%x-%A.out
#SBATCH --partition=gpuq
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{GPU}:1
#SBATCH --cpus-per-task=8
#SBATCH --time={wt}
#SBATCH --mem=100G
{exclude_line}
#SBATCH --export=ALL
set -uo pipefail
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
module load EB5 2>/dev/null || true
cd {REPO}
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export HP_FAST=1
export HP_CACHE_DIR="$PWD/outputs/tensor_cache"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
{llm_env}
ATTEMPT=0
while true; do
  uv run --no-sync python experiments/scaling_hp_search.py \\
    --strategies {strategies} --rounds {ROUNDS} --per_strategy_per_round {PER_ROUND} \\
    --D {D} --ref_only {chr_val_arg} \\
    --reservoir_cache {cache_path} \\
    --data_seed {DATA_SEED} --hp_seed {HP_SEED} \\
    --epochs {EPOCHS} --early_stop_patience {PATIENCE} --min_delta {MIN_DELTA} \\
    --out_dir {od}
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== DONE rc=0 ==="; break; fi
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
    p = Path(f"/tmp/_rsc_{label}.sh")
    p.write_text(script)
    r = subprocess.run([SBATCH, "--parsable", str(p)], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        return r.stdout.strip().split(";")[0], ""
    return None, r.stderr.strip()


def main():
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

    n_sub = n_skip = n_done = n_full = n_nocache = 0
    for reservoir in RESERVOIRS:
        cache_path = pool_cache(reservoir)
        if not Path(cache_path).exists():
            print(f"  SKIP {reservoir}: cache missing ({cache_path}) — generate first")
            n_nocache += 1
            continue
        for D in DS:
            for variant, strategies, model, style in ALL_VARIANTS:
                label = f"rsc_{reservoir}_d{D}_{variant}"
                od = out_dir(reservoir, D, variant)
                if is_complete(od, strategies):
                    n_done += 1
                    continue
                if label in inflight:
                    n_skip += 1
                    continue
                submitted = False
                # V100 backfill: low-priority slow_nice only (grabs idle V100s,
                # 2-day walltime covers a long D=30k cell). H100: normal tiered chain.
                chain = (
                    [("slow_nice", "2-00:00:00")] if GPU == "v100" else qos_chain(D, bool(model))
                )
                for qos, wt in chain:
                    if qcount.get(qos, 0) >= SUBMIT_CAP.get(qos, 0):
                        continue
                    lbl, sh = job_script(
                        reservoir, D, variant, strategies, model, style, cache_path, qos, wt
                    )
                    jid, err = sbatch(lbl, sh)
                    if jid:
                        n_sub += 1
                        qcount[qos] = qcount.get(qos, 0) + 1
                        print(f"  {lbl} -> {jid} [{qos}]")
                        submitted = True
                        break
                    if "QOSMaxSubmitJobPerUser" in err:
                        qcount[qos] = SUBMIT_CAP.get(qos, 0)
                        continue
                    print(f"  ERR {lbl}: {err[:160]}")
                    break
                if not submitted:
                    n_full += 1

    print(
        f"\nSubmitted: {n_sub}  Skipped(inflight): {n_skip}  AlreadyDone: {n_done}  "
        f"AllTiersFull: {n_full}  NoCache: {n_nocache}"
    )
    print(f"Queue counts after run: {qcount}")
    print(
        f"Grid: GPU={GPU} reservoirs={RESERVOIRS} DS={DS} ROUNDS={ROUNDS} "
        f"PER_ROUND={PER_ROUND} (LLM={ROUNDS * PER_ROUND}/cell, algo={3 * ROUNDS * PER_ROUND}/cell)"
    )


if __name__ == "__main__":
    main()
