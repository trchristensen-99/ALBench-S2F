#!/bin/bash
# Run Exp0 (Yeast Scaling, 6M pool) in parallel on Citra GPUs
# Usage: bash scripts/run_citra_parallel.sh
# IMPORTANT: Must be run from the repo root on Citra (~/ALBench-S2F)
# GPU runtime wrapper is required for Citra's older CUDA driver.

mkdir -p logs

COMMON="output_dir=outputs/exp0_yeast_scaling_6m seed=42 wandb_mode=offline"

# GPU 0: 1.0 (Largest – needs most time, gets its own GPU)
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
    fraction=1.0 gpu=0 ${COMMON} > logs/exp0_yeast_1.0.log 2>&1 &
echo "Launched 1.0 on GPU 0"

# GPU 1: 0.5
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
    fraction=0.5 gpu=1 ${COMMON} > logs/exp0_yeast_0.5.log 2>&1 &
echo "Launched 0.5 on GPU 1"

# GPU 2: 0.2
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
    fraction=0.2 gpu=2 ${COMMON} > logs/exp0_yeast_0.2.log 2>&1 &
echo "Launched 0.2 on GPU 2"

# GPU 3: 0.1
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
    fraction=0.1 gpu=3 ${COMMON} > logs/exp0_yeast_0.1.log 2>&1 &
echo "Launched 0.1 on GPU 3"

# GPU 4: 0.05
CUDA_VISIBLE_DEVICES=4 nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
    fraction=0.05 gpu=4 ${COMMON} > logs/exp0_yeast_0.05.log 2>&1 &
echo "Launched 0.05 on GPU 4"

# GPU 5: 0.02
CUDA_VISIBLE_DEVICES=5 nohup bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
    fraction=0.02 gpu=5 ${COMMON} > logs/exp0_yeast_0.02.log 2>&1 &
echo "Launched 0.02 on GPU 5"

# GPU 6: 0.01 then 0.005 (sequential – both are fast)
(
  CUDA_VISIBLE_DEVICES=6 bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
      fraction=0.01 gpu=6 ${COMMON} > logs/exp0_yeast_0.01.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=6 bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
      fraction=0.005 gpu=6 ${COMMON} > logs/exp0_yeast_0.005.log 2>&1
) &
echo "Launched 0.01 then 0.005 on GPU 6"

# GPU 7: 0.002 then 0.001 (sequential – fastest)
(
  CUDA_VISIBLE_DEVICES=7 bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
      fraction=0.002 gpu=7 ${COMMON} > logs/exp0_yeast_0.002.log 2>&1 && \
  CUDA_VISIBLE_DEVICES=7 bash scripts/run_with_runtime.sh python experiments/exp0_yeast_scaling.py \
      fraction=0.001 gpu=7 ${COMMON} > logs/exp0_yeast_0.001.log 2>&1
) &
echo "Launched 0.002 then 0.001 on GPU 7"

echo ""
echo "All 10 fractions launched. Monitor with:"
echo "  tail -f logs/exp0_yeast_*.log"
echo "  watch -n5 nvidia-smi"

