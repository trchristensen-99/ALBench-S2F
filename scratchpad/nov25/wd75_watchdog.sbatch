#!/bin/bash
# wd75: run the overnight watchdog as a CPU batch job.
#
# On cpuq rather than the login node: it wakes every ten minutes for fourteen hours, and a
# sleeping loop that calls squeue is exactly the kind of thing that should not accumulate
# on a shared head node. It holds no GPU.
#SBATCH --job-name=wd75
#SBATCH --partition=cpuq
#SBATCH --account=cshlsbs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=15:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/wd75_%j.log
set -uo pipefail
export TARGET=24 CEILING=12 SLEEP=600 HOURS=14
bash /grid/koo/home/christen/nov25_extra/bin/watchdog.sh
