#!/bin/bash
# Run post-training test-set evaluation for all 10 full-data oracle checkpoints.
# Each task loads outputs/ag_hashfrag_oracle_full/oracle_{task_id}/best_model,
# evaluates on in-distribution / SNV / OOD test sets, and writes test_metrics.json.
# Use this after oracle training times out without reaching the evaluation step.
#
# Submit: sbatch reeval_oracle_alphagenome_full_array.sh
#
#SBATCH --job-name=ag_full_reeval
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-9

source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

export ORACLE_DIR="outputs/ag_hashfrag_oracle_full/oracle_${SLURM_ARRAY_TASK_ID}"
export HEAD_NAME="alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"

echo "Re-evaluating ${ORACLE_DIR} (task ${SLURM_ARRAY_TASK_ID}) on $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python - <<'PYEOF'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from eval_ag import evaluate_hashfrag_test_sets_600bp

task_id = os.environ["SLURM_ARRAY_TASK_ID"]
head_name = os.environ["HEAD_NAME"]
oracle_dir = Path(f"outputs/ag_hashfrag_oracle_full/oracle_{task_id}")
ckpt_dir = oracle_dir / "best_model"
json_path = oracle_dir / "test_metrics.json"

if not (ckpt_dir / "checkpoint").exists():
    print(f"[reeval] ERROR: no checkpoint at {ckpt_dir}", file=sys.stderr)
    sys.exit(1)

# Load existing metadata to preserve seed + best_val_pearson
meta = {}
if json_path.exists():
    with open(json_path) as f:
        meta = json.load(f)

print(f"[reeval] Evaluating oracle_{task_id} from {ckpt_dir} ...")
new_metrics = evaluate_hashfrag_test_sets_600bp(
    ckpt_dir=str(ckpt_dir),
    head_name=head_name,
    data_path="data/k562",
)

meta["test_metrics"] = new_metrics
with open(json_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"[reeval] Wrote {json_path}")
for ts, m in new_metrics.items():
    print(
        f"[reeval]   {ts}: pearson_r={m['pearson_r']:.4f}  "
        f"spearman_r={m['spearman_r']:.4f}  n={m.get('n', '?')}"
    )
PYEOF
