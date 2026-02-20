#!/bin/bash
# Run Exp0 (Yeast Scaling) in parallel on Citra GPUs
# Usage: bash scripts/run_citra_parallel.sh

mkdir -p logs

# Fraction assignments to balance load (approximate)
# Heuristic: Pair largest with smallest, or spread out.
# 8 GPUs available (0-7). 10 fractions.

# GPU 0: 1.0 (Largest)
nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 1.0   --gpu 0 --wandb-mode offline > logs/exp0_yeast_1.0.log 2>&1 &
echo "Launched 1.0 on GPU 0"

# GPU 1: 0.5
nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.5   --gpu 1 --wandb-mode offline > logs/exp0_yeast_0.5.log 2>&1 &
echo "Launched 0.5 on GPU 1"

# GPU 2: 0.2
nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.2   --gpu 2 --wandb-mode offline > logs/exp0_yeast_0.2.log 2>&1 &
echo "Launched 0.2 on GPU 2"

# GPU 3: 0.1
nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.1   --gpu 3 --wandb-mode offline > logs/exp0_yeast_0.1.log 2>&1 &
echo "Launched 0.1 on GPU 3"

# GPU 4: 0.05
nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.05  --gpu 4 --wandb-mode offline > logs/exp0_yeast_0.05.log 2>&1 &
echo "Launched 0.05 on GPU 4"

# GPU 5: 0.02
nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.02  --gpu 5 --wandb-mode offline > logs/exp0_yeast_0.02.log 2>&1 &
echo "Launched 0.02 on GPU 5"

# GPU 6: 0.01 + 0.005 (Small)
# We can run these sequentially on GPU 6
(
  bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.01  --gpu 6 --wandb-mode offline > logs/exp0_yeast_0.01.log 2>&1 && \
  bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.005 --gpu 6 --wandb-mode offline > logs/exp0_yeast_0.005.log 2>&1
) &
echo "Launched 0.01 then 0.005 on GPU 6"

# GPU 7: 0.002 + 0.001 (Smallest)
(
  bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.002 --gpu 7 --wandb-mode offline > logs/exp0_yeast_0.002.log 2>&1 && \
  bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py --fraction 0.001 --gpu 7 --wandb-mode offline > logs/exp0_yeast_0.001.log 2>&1
) &
echo "Launched 0.002 then 0.001 on GPU 7"

echo "All jobs launched in background."
