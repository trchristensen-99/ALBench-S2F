#!/bin/bash
# Auto-submit remaining jobs, retrying every 5 min until successful.
# Run on HPC login node: nohup bash scripts/slurm/auto_submit_remaining.sh > logs/auto_submit5.log 2>&1 &
source /etc/profile.d/modules.sh; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
SB=/cm/shared/apps/slurm/current/bin/sbatch

s1=0; s2=0; s3=0; s4=0; s5=0; s6=0; s7=0; s8=0; s9=0; s10=0; s11=0; s12=0; s13=0; s14=0
for a in $(seq 1 72); do
    echo "=== Attempt $a at $(date) ==="

    if [ $s1 -eq 0 ]; then
        $SB --qos=fast --time=4:00:00 --array=0-7 scripts/slurm/reeval_snv_via_training_scripts.sh 2>/dev/null && s1=1 && echo "  Submitted: foundation SNV re-eval"
    fi

    if [ $s2 -eq 0 ]; then
        TASK=yeast ORACLE=ag STUDENT=dream_rnn $SB --qos=default --time=12:00:00 --array=0-9 scripts/slurm/exp0_oracle_parallel.sh 2>/dev/null && s2=1 && echo "  Submitted: yeast DREAM-RNN x AG"
    fi

    if [ $s3 -eq 0 ]; then
        TASK=yeast ORACLE=ag STUDENT=alphagenome_yeast_s1 $SB --qos=default --time=12:00:00 --array=0-9 scripts/slurm/exp0_oracle_parallel.sh 2>/dev/null && s3=1 && echo "  Submitted: yeast AG S1 x AG"
    fi

    if [ $s4 -eq 0 ]; then
        $SB --array=0-11 scripts/slurm/ag_multicell_3seeds.sh 2>/dev/null && s4=1 && echo "  Submitted: 3-seed AG multicell"
    fi

    if [ $s5 -eq 0 ]; then
        $SB --array=0-15 scripts/slurm/s2_comprehensive_sweep.sh 2>/dev/null && s5=1 && echo "  Submitted: comprehensive S2 sweep"
    fi

    if [ $s6 -eq 0 ]; then
        $SB --array=0-8 scripts/slurm/dream_rnn_single_model.sh 2>/dev/null && s6=1 && echo "  Submitted: single DREAM-RNN"
    fi

    if [ $s7 -eq 0 ]; then
        $SB --qos=fast --time=4:00:00 --array=0-5 scripts/slurm/save_predictions_all_seeds.sh 2>/dev/null && s7=1 && echo "  Submitted: save predictions"
    fi

    if [ $s8 -eq 0 ]; then
        $SB --qos=fast --time=4:00:00 --array=0-3 scripts/slurm/enformer_s2_multicell_sweep.sh 2>/dev/null && s8=1 && echo "  Submitted: Enformer S2 sweep (fixed)"
    fi

    if [ $s9 -eq 0 ]; then
        $SB --array=13-15 scripts/slurm/train_chr_split_all.sh 2>/dev/null && s9=1 && echo "  Submitted: Malinois chr-split"
    fi

    if [ $s10 -eq 0 ]; then
        $SB --array=0-8 scripts/slurm/dream_cnn_real_labels.sh 2>/dev/null && s10=1 && echo "  Submitted: DREAM-CNN real labels"
    fi

    if [ $s11 -eq 0 ]; then
        $SB --array=0-6 scripts/slurm/remaining_gaps.sh 2>/dev/null && s11=1 && echo "  Submitted: remaining gaps"
    fi

    if [ $s12 -eq 0 ]; then
        $SB --array=0-17 scripts/slurm/embedding_strategy_sweep.sh 2>/dev/null && s12=1 && echo "  Submitted: embedding strategy sweep"
    fi

    if [ $s13 -eq 0 ]; then
        $SB --qos=fast --time=4:00:00 --array=0-2 scripts/slurm/reeval_k562_real_labels.sh 2>/dev/null && s13=1 && echo "  Submitted: K562 real-label re-eval"
    fi

    if [ $s14 -eq 0 ]; then
        $SB --array=0-1 scripts/slurm/fix_remaining_anomalies.sh 2>/dev/null && s14=1 && echo "  Submitted: AG SNV + Borzoi OOD fixes"
    fi

    t=$((s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13 + s14))
    if [ $t -eq 14 ]; then
        echo "All 14 jobs submitted!"
        break
    fi
    echo "  $t/14 submitted, waiting 5 min..."
    sleep 300
done
echo "Done at $(date)"
