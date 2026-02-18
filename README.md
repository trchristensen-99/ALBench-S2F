# ALBench-S2F

**Active Learning Benchmark for Sequence-to-Function Prediction**

Compare reservoir sampling strategies and acquisition functions for training genomic sequence-to-function models on MPRA datasets.

## 🚀 Quick Start (Production Environments)

### 1. Citra (Dev/GPU Server)
*Use for: Interactive development, smaller experiments, debugging.*

**Connect:**
```bash
ssh trevor@143.48.59.3  # (Use your key)
cd ~/ALBench-S2F
```

**Run Experiment 0 (Yeast Scaling) - Managed Queue:**
The managed script avoids OOM errors by intelligently scheduling fractions on free GPUs.
```bash
# Start the manager (runs in background)
nohup python3 scripts/run_citra_managed.py > logs/citra_managed.log 2>&1 & disown

# Monitor progress
tail -f logs/citra_managed.log
```

**Run HashFrag (K562):**
```bash
nohup bash scripts/run_with_runtime.sh python scripts/create_hashfrag_splits.py \
    --data-dir data/k562 --threshold 60 > logs/hashfrag_k562.log 2>&1 &
```

---

### 2. CSHL HPC (Cluster)
*Use for: Full-scale experiments, long-running jobs.*

**Connect:**
```bash
ssh christen@bamdev4.cshl.edu
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
source setup_env.sh
```

**Run Experiment 0 (Yeast Scaling) - Resumable Slurm Array:**
The script auto-checkpoints `last_model.pt` every epoch. If it hits the 12h limit, just resubmit it to resume.

**Option A: Manual Submission**
```bash
sbatch scripts/slurm/exp0_yeast_scaling.sh
```

**Option B: Automated Watchdog (Recommended)**
Runs on login node, monitors completion, and auto-resubmits if jobs finish/timeout without completing all work.
```bash
nohup python3 scripts/slurm/watchdog_exp0.py > logs/watchdog_exp0.log 2>&1 &
```

**Run HashFrag (K562):**
```bash
sbatch scripts/slurm/create_hashfrag_splits.sh
```
*Note: Fits safely within 12h limit.*

---

## 🛠️ Local Development (Mac)

**Installation:**
```bash
uv sync --extra dev
uv run pre-commit install
```

**Tests:**
```bash
uv run pytest tests/ -v
```

## 📊 Monitoring

- **Logs (Both):** Check `logs/` directory.
- **W&B:** Runs log to `albench-s2f` project.
- **HPC Status:** `squeue -u christen`
- **Citra Status:** `nvidia-smi` or `pgrep -f exp0`

## 📦 Project Layout

- `experiments/`: Main entry points (`exp0_yeast_scaling.py`, etc.)
- `scripts/`: Helper scripts (setup, runners, Slurm templates)
- `albench/`: Core library code
