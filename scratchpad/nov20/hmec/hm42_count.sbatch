#!/bin/bash
# hm42: count HMEC MPRA reads per designed oligo -- 5 RNA replicates and 5 plasmid controls.
#
# R1 only. R2 reads the same fragment from the other end, so counting it too would count
# each molecule twice; it is held in reserve as a way to rescue fragments whose R1 carried
# a sequencing error, not as extra depth.
#
# One array task per file, CPU only: the work is a gzip stream and a vectorised hash lookup,
# ~33 s per million reads, so a 5 GB file is about 20 minutes.
#SBATCH --job-name=hm42
#SBATCH --partition=cpuq
#SBATCH --account=cshlsbs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/hmec/logs/hm42_%A_%a.log
#SBATCH --array=0-9
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p outputs/hmec/counts

FQ=(data/external/hmec_mpra/fastq/rna_rep1_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/rna_rep2_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/rna_rep3_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/rna_rep4_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/rna_rep5_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/dna_rep1_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/dna_rep2_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/dna_rep3_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/dna_rep4_R1_*.fastq.gz
    data/external/hmec_mpra/fastq/dna_rep5_R1_*.fastq.gz)
TAG=(rna_rep1 rna_rep2 rna_rep3 rna_rep4 rna_rep5
     dna_rep1 dna_rep2 dna_rep3 dna_rep4 dna_rep5)

f=${FQ[$SLURM_ARRAY_TASK_ID]}
t=${TAG[$SLURM_ARRAY_TASK_ID]}
echo "counting $t <- $f"
.venv/bin/python -I scratchpad/nov20/hmec/count_oligos.py \
  --design data/external/tewhey_elements/ENCSR091QEV__ENCFF770UJN.fa \
           data/external/tewhey_elements/ENCSR935CNL__ENCFF090JTW.fa \
  --fastq "$f" --out "outputs/hmec/counts/${t}.npz"
