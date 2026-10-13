#!/bin/bash
#SBATCH --job-name=sc_watchdog
#SBATCH --partition=cpuq --qos=slow_nice --cpus-per-task=1 --mem=4G
#SBATCH --time=2-00:00:00 --requeue
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/scaling_canonical_oct13/logs/sc_watchdog_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
exec bash /grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/scaling_canonical_oct13/sc_watchdog.sh
