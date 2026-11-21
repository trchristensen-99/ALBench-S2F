#!/bin/bash
# cs41: an AutoResearch-style loop whose objective is CROSS-CELL generalisation.
#
# The architecture search optimises held-out sequence in one cell. This optimises the thing
# ct41 and ct42 say we cannot do: predicting a cell the model never trained on, from a
# written description of it. The proposer writes `Net(d_cond)` with `forward(x, z)`, so it
# searches the conditioning MECHANISM as well as the trunk -- which is where the bottleneck
# is, since conditioning already pays (+0.03 genomic over cell-blind) while being
# indistinguishable from permuted descriptions.
#
# Every proposal is trained TWICE, real descriptions and deranged, and the margin between
# them is the number that counts. The control runs on every model rather than on the
# winner, because otherwise the loop selects for designs that use the vector as a cell ID
# all the way through and the control at the end just tells us so.
#
# H100 default QOS: the V100 slow_nice slots are full with ct43 and ye42, and these H100s
# are idle. Pilot: 6 rounds x 3 proposals x 2 arms, ~10 min per arm = ~6 GPU-h.
#SBATCH --job-name=cs41
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/celltype/logs/cs41_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/celltype/logs

.venv/bin/python -m albench.celltype.search \
  --rounds 6 --per-round 3 \
  --epochs 8 --batch-size 512 --lr 0.005 \
  --subset 600000 --max-params 4000000 --max-sec 900 \
  --seed 0 --out outputs/celltype/cs41
