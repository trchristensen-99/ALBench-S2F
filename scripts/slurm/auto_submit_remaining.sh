#!/bin/bash
# Auto-submit remaining jobs, retrying every 5 min until successful.
# Run on HPC login node: nohup bash scripts/slurm/auto_submit_remaining.sh > logs/auto_submit5.log 2>&1 &
source /etc/profile.d/modules.sh; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
SB=/cm/shared/apps/slurm/current/bin/sbatch

s1=0; s2=0; s3=0; s4=0
for a in $(seq 1 48); do
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

    t=$((s1 + s2 + s3 + s4))
    if [ $t -eq 4 ]; then
        echo "All 4 jobs submitted!"
        break
    fi
    echo "  $t/4 submitted, waiting 5 min..."
    sleep 300
done
echo "Done at $(date)"
