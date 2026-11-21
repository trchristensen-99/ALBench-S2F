#!/bin/bash
# hm41: pull the HMEC primary-cell MPRA reads (RNA) and their plasmid control (DNA).
#
# This is the only non-immortalised biosample in the corpus, so it is the only place the
# `immortalised`/`malignant` fields in the cell descriptions can vary. ENCODE released reads
# but no quantification and no barcode-to-oligo map, so activity has to be counted from the
# fastqs; this job only fetches them.
#
# cpuq, not the login node: 107 GB of network I/O. One array task per file so the 20
# transfers run concurrently and a failed one can be re-queued alone.
#SBATCH --job-name=hm41
#SBATCH --partition=cpuq
#SBATCH --account=cshlsbs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/hmec/logs/hm41_%A_%a.log
#SBATCH --array=0-19%10
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
mkdir -p outputs/hmec/logs data/external/hmec_mpra/fastq
.venv/bin/python -I scratchpad/nov20/hmec/fetch_hmec_fastq.py \
  --manifest data/external/hmec_mpra/MANIFEST.json \
  --out data/external/hmec_mpra/fastq \
  --index "$SLURM_ARRAY_TASK_ID"
