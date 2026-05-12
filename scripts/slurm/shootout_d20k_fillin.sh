#!/bin/bash
#SBATCH --job-name=shootout_d20k_fillin
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=03:30:00
#SBATCH --mem=120G

# Fill-in for the 6 shootout configs that died from the disk-full crisis
# (low_lr_bs256, high_lr_bs1024 was done, wider_arch, deeper_arch,
# with_shift_aug, low_dropout_aggressive) plus a couple of new variants
# targeting the legnet_published_default's neighborhood.

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1  # avoid compile hang on tiny LegNet
export HP_CACHE_DIR=$PWD/outputs/tensor_cache
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

OUT=results/preflight/shootout_d20k_fillin
rm -rf "$OUT"
mkdir -p "$OUT"

cat > "$OUT/configs.json" <<'EOF'
[
  {"label":"low_lr_bs256",          "arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rev_complement",
   "output_dir":"results/preflight/shootout_d20k_fillin/low_lr_bs256",
   "hp_overrides":["lr=0.001","batch_size=256","weight_decay=0.1","dropout=0.0","block_sizes=[256,256,128,128,64,64,32,32]","ks=5"]},
  {"label":"wider_arch",            "arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rev_complement",
   "output_dir":"results/preflight/shootout_d20k_fillin/wider_arch",
   "hp_overrides":["lr=0.003","batch_size=512","weight_decay=0.05","dropout=0.0","block_sizes=[512,512,256,256,128,128,64,64]","ks=5"]},
  {"label":"deeper_arch",           "arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rev_complement",
   "output_dir":"results/preflight/shootout_d20k_fillin/deeper_arch",
   "hp_overrides":["lr=0.003","batch_size=512","weight_decay=0.1","dropout=0.0","block_sizes=[256,256,128,128,128,64,64,64,32]","ks=5"]},
  {"label":"with_shift_aug",        "arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rc_shift",
   "output_dir":"results/preflight/shootout_d20k_fillin/with_shift_aug",
   "hp_overrides":["lr=0.005","batch_size=1024","weight_decay=0.1","dropout=0.0","block_sizes=[256,256,128,128,64,64,32,32]","ks=5"]},
  {"label":"low_dropout_aggressive","arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rev_complement",
   "output_dir":"results/preflight/shootout_d20k_fillin/low_dropout_aggressive",
   "hp_overrides":["lr=0.005","batch_size=512","weight_decay=0.05","dropout=0.05","block_sizes=[256,256,128,128,64,64,32,32]","ks=5"]},
  {"label":"plus_lr_2x_bs1024",     "arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rev_complement",
   "output_dir":"results/preflight/shootout_d20k_fillin/plus_lr_2x_bs1024",
   "hp_overrides":["lr=0.01","batch_size=1024","weight_decay=0.1","dropout=0.0","block_sizes=[256,256,128,128,64,64,32,32]","ks=5"]},
  {"label":"plus_higher_wd",        "arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rev_complement",
   "output_dir":"results/preflight/shootout_d20k_fillin/plus_higher_wd",
   "hp_overrides":["lr=0.005","batch_size=1024","weight_decay=0.2","dropout=0.0","block_sizes=[256,256,128,128,64,64,32,32]","ks=5"]},
  {"label":"plus_dropout_0.1",      "arch":"legnet","d_train":20000,"seed":42,"epochs":80,"patience":15,"aug":"rev_complement",
   "output_dir":"results/preflight/shootout_d20k_fillin/plus_dropout_0.1",
   "hp_overrides":["lr=0.005","batch_size=1024","weight_decay=0.1","dropout=0.1","block_sizes=[256,256,128,128,64,64,32,32]","ks=5"]}
]
EOF

# HP_FAST=1 makes parallel_gpu_runner pass --fast to each trial (sans compile)
export HP_FAST=1
uv run --no-sync python scripts/preflight/parallel_gpu_runner.py "$OUT/configs.json" 4 2>&1 | tee "$OUT/driver.log"
