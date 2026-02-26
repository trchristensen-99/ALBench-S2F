#!/bin/bash
# Re-evaluate all 10 hashFrag oracle checkpoints with corrected snv_abs metric.
# Patches test_metrics.json in-place, preserving seed/best_val_pearson metadata.
# Submit: sbatch reeval_oracle_hashfrag_array.sh
#
#SBATCH --job-name=ag_hf_reeval
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-9

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer="

ORACLE_DIR="outputs/ag_hashfrag_oracle/oracle_${SLURM_ARRAY_TASK_ID}"
echo "Re-evaluating ${ORACLE_DIR} with fixed snv_abs metric..."

uv run python - <<'PYEOF'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from eval_ag import evaluate_hashfrag_test_sets_600bp

task_id = os.environ["SLURM_ARRAY_TASK_ID"]
oracle_dir = Path(f"outputs/ag_hashfrag_oracle/oracle_{task_id}")
ckpt_dir = oracle_dir / "best_model"
json_path = oracle_dir / "test_metrics.json"
head_name = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"

if not (ckpt_dir / "checkpoint").exists():
    print(f"[reeval] ERROR: no checkpoint at {ckpt_dir}", file=sys.stderr)
    sys.exit(1)

# Load existing JSON to preserve seed + best_val_pearson
meta = {}
if json_path.exists():
    with open(json_path) as f:
        meta = json.load(f)

print(f"[reeval] Running eval for oracle_{task_id}...")
new_metrics = evaluate_hashfrag_test_sets_600bp(
    ckpt_dir=str(ckpt_dir),
    head_name=head_name,
    data_path="data/k562",
)

meta["test_metrics"] = new_metrics
with open(json_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"[reeval] Patched {json_path}")
for ts, m in new_metrics.items():
    print(f"[reeval]   {ts}: pearson_r={m['pearson_r']:.4f}  spearman_r={m['spearman_r']:.4f}")
PYEOF
