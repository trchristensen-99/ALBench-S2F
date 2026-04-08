# Job Submission Order

## Phase 0: Data Preparation (no GPU, run first)

```bash
# Create test set TSV files for HepG2/SK-N-SH (in-dist + SNV)
/cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/create_cellline_test_sets.sh

# Build OOD designed test sets for HepG2/SK-N-SH (needs Zenodo OL46 data)
/cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/build_multicell_ood_data.sh
```

## Phase 1: Bar Plot Experiments (real labels)

After Phase 0 completes:

```bash
# DREAM-RNN on HepG2 + SK-N-SH (3 seeds each, ground_truth labels)
/cm/shared/apps/slurm/current/bin/sbatch --array=0-5 scripts/slurm/dream_rnn_multicell_v2.sh
```

## Phase 2: OOD Evaluation (after Phase 1 training + Phase 0 OOD data)

```bash
# Evaluate all existing multicell checkpoints on OOD test sets
/cm/shared/apps/slurm/current/bin/sbatch --array=0-15 scripts/slurm/eval_multicell_ood.sh
```

## Phase 3: Scaling Experiments (oracle labels)

Can run in parallel with Phases 1-2:

```bash
# K562 DREAM-RNN remaining sizes (32k, 64k, 160k, 320k)
TASK=k562 /cm/shared/apps/slurm/current/bin/sbatch --array=0-3 scripts/slurm/exp0_dream_rnn_remaining.sh

# Yeast DREAM-RNN remaining sizes (303k-6.1M)
TASK=yeast /cm/shared/apps/slurm/current/bin/sbatch --array=0-4 scripts/slurm/exp0_dream_rnn_remaining.sh

# Yeast AlphaGenome S2 all sizes (0 results currently)
/cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/exp0_yeast_ag_s2.sh
```

## Phase 4: Cross-Oracle Scaling (optional, can run anytime)

```bash
# K562 with DREAM-RNN oracle (default is AG)
TASK=k562 ORACLE=dream_rnn STUDENT=dream_cnn /cm/shared/apps/slurm/current/bin/sbatch --array=0-6 scripts/slurm/exp0_oracle_parallel.sh
TASK=k562 ORACLE=dream_rnn STUDENT=dream_rnn /cm/shared/apps/slurm/current/bin/sbatch --array=0-6 scripts/slurm/exp0_oracle_parallel.sh
TASK=k562 ORACLE=dream_rnn STUDENT=alphagenome_k562_s1 /cm/shared/apps/slurm/current/bin/sbatch --array=0-6 scripts/slurm/exp0_oracle_parallel.sh

# Yeast with AG oracle (default is DREAM-RNN)
TASK=yeast ORACLE=ag STUDENT=dream_cnn /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
TASK=yeast ORACLE=ag STUDENT=dream_rnn /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
TASK=yeast ORACLE=ag STUDENT=alphagenome_yeast_s1 /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/exp0_oracle_parallel.sh
```
