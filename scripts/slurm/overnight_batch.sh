#!/bin/bash
# Overnight batch: fill all remaining Exp0 gaps using default/fast QoS.
# These are supplementary to the slow_nice jobs already running.
#
# Array tasks (14 total):
#   --- Real-label gaps (ground_truth) ---
#   0: DREAM-CNN ground_truth extended (n=296382) — genomic reservoir
#   1: DREAM-RNN ground_truth extended (n=159871 more seeds + n=296382)
#   2: AG S1 ground_truth (all sizes 3197-296382) — ENTIRELY MISSING
#   --- Cross-oracle gaps ---
#   3: LegNet + AG S2 oracle (duplicate on default for faster progress)
#   4: LegNet + LegNet oracle (duplicate on default)
#   --- DREAM-CNN with different oracles ---
#   5: DREAM-CNN + AG oracle (all sizes, genomic) — fills v4 gaps
#   6: DREAM-CNN + DREAM-RNN oracle extended (n=296382)
#   --- DREAM-RNN with AG oracle (genomic) ---
#   7: DREAM-RNN + AG oracle (genomic, all sizes)
#   --- Augmentation experiments on LegNet ---
#   8: LegNet aug: RC+shift (chr-split, ground_truth)
#   9: LegNet aug: alt_alleles (chr-split, ground_truth)
#   10: LegNet aug: duplication (chr-split, ground_truth)
#   11: LegNet aug: alt+shift (chr-split, ground_truth)
#   --- Enformer S2 fix (needs H100, batch_size=32 to avoid OOM) ---
#   12: Enformer S2 fix HepG2 seeds 1,2
#   13: Enformer S2 fix SknSh seeds 1,2
#
# Submit on default + fast (15 GPU slots available):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-11 --qos=default --time=12:00:00 scripts/slurm/overnight_batch.sh
#   /cm/shared/apps/slurm/current/bin/sbatch --array=12-13 --qos=default --time=04:00:00 --gres=gpu:h100:1 --mem=200G scripts/slurm/overnight_batch.sh
#
#SBATCH --job-name=overnight
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== overnight batch task=${T} node=${SLURMD_NODENAME} $(date) ==="

case ${T} in

# ── Real-label (ground_truth) gaps ──────────────────────────────────────

0)
    echo "DREAM-CNN ground_truth extended (n=296382)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_cnn --oracle ground_truth \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/dream_cnn_ground_truth \
        --training-sizes 296382 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10 --save-predictions
    ;;

1)
    echo "DREAM-RNN ground_truth extended"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn --oracle ground_truth \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/dream_rnn_ground_truth \
        --training-sizes 159871 296382 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10 --save-predictions
    ;;

2)
    echo "AG S1 ground_truth (all sizes) — NEW"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student alphagenome_k562_s1 --oracle ground_truth \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/alphagenome_k562_s1_ground_truth \
        --training-sizes 3197 6395 15987 31974 63949 159871 296382 \
        --epochs 50 --early-stop-patience 7 --no-hp-sweep \
        --save-predictions
    ;;

# ── Cross-oracle (duplicate on default for speed) ──────────────────────

3)
    echo "LegNet + AG S2 oracle (genomic, all sizes)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ag_s2 \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/legnet_oracle_ag_s2 \
        --training-sizes 3197 6395 15987 31974 63949 159871 296382 \
        --lr 0.001 --batch-size 1024 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions || true
    ;;

4)
    echo "LegNet + LegNet oracle (genomic, all sizes)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle legnet \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/legnet_oracle_legnet \
        --training-sizes 3197 6395 15987 31974 63949 159871 296382 \
        --lr 0.001 --batch-size 1024 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions || true
    ;;

# ── DREAM-CNN cross-oracle ─────────────────────────────────────────────

5)
    echo "DREAM-CNN + AG oracle (genomic, fill gaps)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_cnn --oracle ag \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/dream_cnn \
        --training-sizes 3197 6395 15987 31974 63949 159871 296382 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions || true
    ;;

6)
    echo "DREAM-CNN + DREAM-RNN oracle extended"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_cnn --oracle dream_rnn \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/dream_cnn_oracle_dream_rnn \
        --training-sizes 296382 --epochs 80 --ensemble-size 1 \
        --early-stop-patience 10 --save-predictions || true
    ;;

# ── DREAM-RNN cross-oracle ────────────────────────────────────────────

7)
    echo "DREAM-RNN + AG oracle (genomic, all sizes)"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student dream_rnn --oracle ag \
        --reservoir genomic --n-replicates 3 --seed 42 \
        --output-dir outputs/exp0_oracle_scaling_v4/k562/dream_rnn \
        --training-sizes 3197 6395 15987 31974 63949 159871 296382 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions || true
    ;;

# ── LegNet augmentation experiments ────────────────────────────────────

8)
    echo "LegNet aug: RC+shift"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ground_truth \
        --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 --no-hp-sweep \
        --lr 0.001 --batch-size 1024 \
        --output-dir outputs/aug_comparison/legnet/rc_shift \
        --training-sizes 400000 \
        --shift-aug --max-shift 15 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions
    ;;

9)
    echo "LegNet aug: alt_alleles"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ground_truth \
        --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 --no-hp-sweep \
        --lr 0.001 --batch-size 1024 \
        --output-dir outputs/aug_comparison/legnet/alt_alleles \
        --training-sizes 400000 \
        --include-alt-alleles \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions
    ;;

10)
    echo "LegNet aug: duplication"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ground_truth \
        --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 --no-hp-sweep \
        --lr 0.001 --batch-size 1024 \
        --output-dir outputs/aug_comparison/legnet/duplication \
        --training-sizes 400000 \
        --duplication-cutoff 0.5 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions
    ;;

11)
    echo "LegNet aug: alt+shift"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student legnet --oracle ground_truth \
        --reservoir genomic --chr-split \
        --n-replicates 3 --seed 42 --no-hp-sweep \
        --lr 0.001 --batch-size 1024 \
        --output-dir outputs/aug_comparison/legnet/alt_shift \
        --training-sizes 400000 \
        --include-alt-alleles --shift-aug --max-shift 15 \
        --epochs 80 --ensemble-size 1 --early-stop-patience 10 \
        --save-predictions
    ;;

# ── Enformer S2 fix (needs H100 + large memory) ───────────────────────

12)
    echo "Enformer S2 fix HepG2 seeds 1,2"
    for SEED in 1 2; do
        OUT="outputs/enformer_hepg2_stage2/seed_${SEED}"
        S1="outputs/enformer_hepg2_cached/seed_${SEED}/seed_${SEED}"
        [ -f "${OUT}/result.json" ] && [ ! -f "${OUT}/result_WRONG_k562.json" ] && continue
        [ -f "${OUT}/result.json" ] && mv "${OUT}/result.json" "${OUT}/result_WRONG_k562.json" 2>/dev/null
        [ -f "${OUT}/best_model.pt" ] && mv "${OUT}/best_model.pt" "${OUT}/best_model_WRONG_k562.pt" 2>/dev/null
        echo "  Seed ${SEED}..."
        uv run --no-sync python experiments/train_foundation_stage2.py \
            ++model_name=enformer ++data_path="data/hepg2" ++cell_line=hepg2 \
            ++chr_split=True ++seed=${SEED} ++output_dir="${OUT}" \
            ++stage1_result_dir="${S1}" \
            ++encoder_lr=1e-4 ++head_lr=1e-3 ++epochs=15 \
            ++early_stop_patience=5 ++unfreeze_mode=transformer \
            ++batch_size=32 ++save_encoder=True ++amp_mode=bfloat16 || true
    done
    ;;

13)
    echo "Enformer S2 fix SknSh seeds 1,2"
    for SEED in 1 2; do
        OUT="outputs/enformer_sknsh_stage2/seed_${SEED}"
        S1="outputs/enformer_sknsh_cached/seed_${SEED}/seed_${SEED}"
        [ -f "${OUT}/result.json" ] && [ ! -f "${OUT}/result_WRONG_k562.json" ] && continue
        [ -f "${OUT}/result.json" ] && mv "${OUT}/result.json" "${OUT}/result_WRONG_k562.json" 2>/dev/null
        [ -f "${OUT}/best_model.pt" ] && mv "${OUT}/best_model.pt" "${OUT}/best_model_WRONG_k562.pt" 2>/dev/null
        echo "  Seed ${SEED}..."
        uv run --no-sync python experiments/train_foundation_stage2.py \
            ++model_name=enformer ++data_path="data/sknsh" ++cell_line=sknsh \
            ++chr_split=True ++seed=${SEED} ++output_dir="${OUT}" \
            ++stage1_result_dir="${S1}" \
            ++encoder_lr=1e-4 ++head_lr=1e-3 ++epochs=15 \
            ++early_stop_patience=5 ++unfreeze_mode=transformer \
            ++batch_size=32 ++save_encoder=True ++amp_mode=bfloat16 || true
    done
    ;;

esac

echo "=== task=${T} DONE — $(date) ==="
