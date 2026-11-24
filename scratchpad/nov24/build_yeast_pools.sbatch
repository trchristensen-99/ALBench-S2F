#!/bin/bash
# ye46: the first yeast reservoir pools. Half the deliverable has no data behind it.
#
# The collaborators will order 4M yeast sequences as well as 4M human ones, and every
# reservoir comparison this project has run is K562. There is no yeast equivalent of Arm
# A, so a yeast allocation recommendation would currently be an opinion.
#
# This builds the pools, which is the half that needs no new code: the samplers already
# take task="yeast" (80-bp variable region inside the fixed DREAM promoter context) and
# build_pool.py labels with the DREAM-RNN ensemble, which as of today resolves to the
# ye43 rebuild rather than the truncated six-epoch folds.
#
# What it does NOT do is train yeast students. bench33_legnet.py is the K562 harness --
# 200-bp insert, 240-bp standard, real plasmid flanks -- and a yeast student needs its
# own. That is the next piece of work and it is the thing standing between us and a yeast
# recommendation; building the pools first means the harness arrives to data rather than
# the other way round.
#
# Even before students, the labelled pools answer something: whether the activity
# distribution each reservoir induces in yeast resembles the one it induces in K562. If
# motif planting shifts yeast activity the way it shifts K562 activity, the human result
# transfers as a prior; if it does not, yeast needs its own ranking from scratch.
#SBATCH --job-name=ye46
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/grid_pools/logs/ye46_%A_%a.log
#SBATCH --array=0-4
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/grid_pools/logs outputs/grid_pools/yeast_raw

# Only the strategies that GENERATE rather than derive. mutagenesis, evoaug, gc_matched
# and dinuc_shuffle all perturb or match a candidate pool, and the only pool wired up is
# the human genomic background -- building them here would silently perturb human
# sequence and drop an 80-bp fragment of a human CRE into the yeast promoter context.
# build_pool.py now refuses that outright rather than substituting. A yeast `genomic` arm
# (native S. cerevisiae promoters) is the asset to add next, and it is also the arm a
# yeast comparison most needs.
STRATS=(random random_gc_sweep motif_shared_core motif_ct_enriched motif_syntax_core)
i=$SLURM_ARRAY_TASK_ID
S=${STRATS[$i]}

echo "[$(date)] ye46 task $i: strategy=$S task=yeast"
uv run --no-sync python scripts/build_pool.py \
  --strategy "$S" \
  --size 30000 \
  --generation-seed 42 \
  --task yeast \
  --oracle dream_rnn \
  --oracle-id yeast_dream_rnn_ye43_10fold \
  --out-dir outputs/grid_pools/yeast_raw
echo "[$(date)] done $S"
