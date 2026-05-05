#!/bin/bash
# Score every Task 2 best.pt checkpoint against the 13 eval-set panels
# under outputs/eval_sets_expanded/, plus the in_dist + ood + snv test sets.
# 36 ckpts × 14 panels = 504 quick inferences (each <30s on H100).
# Submitted as a 36-task array; each task scores all 14 panels for one ckpt.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# Build list of Task 2 ckpts
mapfile -t CKPTS < <(find "$REPO/results/preflight/task2_d_min" -name 'best.pt' 2>/dev/null | sort)
N=${#CKPTS[@]}
if [ "$N" -eq 0 ]; then
    echo "No Task 2 ckpts found at $REPO/results/preflight/task2_d_min"
    exit 1
fi
echo "Found $N Task 2 ckpts to score"

# Write the array script
cat > /tmp/_pf_task2_score.sh <<EOF
#!/bin/bash
#SBATCH --job-name=pf_task2_score
#SBATCH --output=$REPO/logs/%x-%A-%a.out
#SBATCH --error=$REPO/logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --array=0-$((N - 1))%4
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd $REPO || exit 1
export PYTHONPATH="\$PWD"
source scripts/slurm/setup_hpc_deps.sh

CKPTS=($(printf '"%s" ' "${CKPTS[@]}"))
CKPT="\${CKPTS[\$SLURM_ARRAY_TASK_ID]}"

# Infer arch from path: results/preflight/task2_d_min/<arch>/d<D>/seed<S>/best.pt
ARCH=\$(echo "\$CKPT" | grep -oE '/(legnet|dream_rnn|dream_attn)/' | head -1 | tr -d '/')
echo "Scoring \$CKPT (arch=\$ARCH)"
uv run --no-sync python scripts/preflight/score_eval_sets.py \\
    --ckpt "\$CKPT" --arch "\$ARCH"
EOF

JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable /tmp/_pf_task2_score.sh)
rm -f /tmp/_pf_task2_score.sh
echo "Submitted as array job $JOB (max 4 concurrent on fast queue)"
echo "Per-ckpt outputs land at <ckpt_dir>/eval_sets_panel.json"
