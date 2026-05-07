#!/bin/bash
# Pre-flight finalization-time validation sweep.
#
# Three small experiments asked by user:
#   (A) LegNet aug-fairness confirm: re-test rev_complement / rc_shift
#       at the new locked HPs (lr=3e-3, bs=512). If they're still ~2x
#       worse than aug=none, our claim that "aug hurts LegNet" is solid.
#   (B) dream_rnn dropout edge extension: test dropout_lstm = 0.05, 0.10
#       at locked HPs. Current lock is 0.15 (lower edge of [0.15, 0.30, 0.60]).
#   (C) HP universality across D: at LOCKED HPs (post-retry), train each
#       arch at d ∈ {500, 30000, 600000} × 2 seeds = 18 cells. Verifies
#       locked HPs work across the scaling-sweep N range. If results look
#       bad at small N, we need per-N HP tuning.
#
# Total: 4 (A) + 4 (B) + 18 (C) = 26 cells. Distributed across all
# 3 queues for max parallel throughput. ETA ~2-3 hr.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# ---------------------------------------------------------------
# Helper: submit one cell via launch.sh
# Args: arch d_train seed aug epochs out_dir job_name qos time hp_overrides...
# ---------------------------------------------------------------
submit_cell() {
    local arch=$1 d=$2 seed=$3 aug=$4 epochs=$5 out=$6 jname=$7 qos=$8 t=$9
    shift 9
    local hps=("$@")
    if [ -f "$out/result.json" ]; then
        echo "  [skip] $jname — done"
        return 0
    fi
    if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
        | grep -qx "$jname"; then
        echo "  [skip] $jname already in queue"
        return 0
    fi
    PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
    PREFLIGHT_EPOCHS=$epochs \
    PREFLIGHT_EARLY_STOP_PATIENCE=15 \
    PREFLIGHT_AUG=$aug \
    PREFLIGHT_LABEL_SOURCE=ag_oracle \
    PREFLIGHT_OUT=$out \
    PREFLIGHT_JOB_NAME=$jname \
        bash scripts/preflight/launch.sh "$arch" "$d" "$seed" "${hps[@]}"
}

n=0

# ============================================================
# (A) LegNet aug-fairness confirm at retry-best HPs
# ============================================================
echo "=== (A) LegNet aug confirm at lr=3e-3 bs=512 ==="
for aug in rev_complement rc_shift; do
    for seed in 42 123; do
        out="results/preflight/task5_legnet_aug_confirm/${aug}/seed${seed}"
        jname="pf_legnetfx_${aug}_s${seed}"
        # Route to fast since these are short
        submit_cell legnet 600000 $seed "$aug" 35 "$out" "$jname" \
            slow_nice 12:00:00 "lr=0.003" "batch_size=512"
        n=$((n + 1))
    done
done

# ============================================================
# (B) dream_rnn dropout edge extension (test 0.05, 0.10)
# ============================================================
echo "=== (B) dream_rnn dropout edge extension ==="
for dr in 0.05 0.10; do
    for seed in 42 123; do
        # dropout_lstm is the relevant key for dream_rnn
        out="results/preflight/task7_dream_rnn_dropout_ext/dropout_lstm_${dr}/seed${seed}"
        jname="pf_drnndr_${dr/./p}_s${seed}"
        submit_cell dream_rnn 600000 $seed rev_complement 66 "$out" "$jname" \
            slow_nice 12:00:00 "lr=0.003" "batch_size=256" "dropout_lstm=$dr"
        n=$((n + 1))
    done
done

# ============================================================
# (C) HP universality across D — LOCKED HPs at 3 D values × 2 seeds
# ============================================================
echo "=== (C) HP universality (locked HPs at d ∈ {500, 30000, 600000}) ==="

# Per-arch locked HP packs
declare -A LR=( [legnet]=0.003 [dream_rnn]=0.003 [dream_attn]=0.0003 )
declare -A BS=( [legnet]=512 [dream_rnn]=256 [dream_attn]=64 )
declare -A AUG=( [legnet]=none [dream_rnn]=rev_complement [dream_attn]=rc_shift )
declare -A DROPOUT_KEY=( [legnet]=dropout [dream_rnn]=dropout_lstm [dream_attn]=core_dropout )
declare -A DROPOUT_VAL=( [legnet]=0.0 [dream_rnn]=0.15 [dream_attn]=0.1 )
declare -A EP=( [legnet]=35 [dream_rnn]=66 [dream_attn]=108 )

for arch in legnet dream_rnn dream_attn; do
    for d in 500 30000 600000; do
        for seed in 42 123; do
            out="results/preflight/task_hp_universality/${arch}/d${d}/seed${seed}"
            jname="pf_univ_${arch}_d${d}_s${seed}"
            # Route everything to slow_nice (fast/default already saturated by task9)
            qos=slow_nice; t=12:00:00
            dr_key="${DROPOUT_KEY[$arch]}"
            dr_val="${DROPOUT_VAL[$arch]}"
            submit_cell "$arch" "$d" "$seed" "${AUG[$arch]}" "${EP[$arch]}" "$out" "$jname" \
                "$qos" "$t" "lr=${LR[$arch]}" "batch_size=${BS[$arch]}" "${dr_key}=${dr_val}"
            n=$((n + 1))
        done
    done
done

echo
echo "=== Submitted $n cells total ==="
echo "Outputs:"
echo "  (A) results/preflight/task5_legnet_aug_confirm/<aug>/seed<n>/"
echo "  (B) results/preflight/task7_dream_rnn_dropout_ext/dropout_lstm_<v>/seed<n>/"
echo "  (C) results/preflight/task_hp_universality/<arch>/d<d>/seed<n>/"
