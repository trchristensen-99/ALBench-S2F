#!/bin/bash
# cs42: cs41 again, with the LLM credentials it was missing.
#
# cs41 trained its cell-blind floor (r = 0.7784 on HCT116) and then failed every one of six
# proposer calls with "Could not resolve authentication method". The transport picks the CLI
# path only when CLAUDE_CODE_OAUTH_TOKEN is set and otherwise falls through to the API,
# which has no key in a batch job; the dispatcher's own LLM cells source this file and ours
# did not.
#SBATCH --job-name=cs42
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/celltype/logs/cs42_%j.log
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
set +u; . "$HOME/.config/albench/llm.env"; set -u
export LLM_MODEL=claude-sonnet-5-5
mkdir -p outputs/celltype/logs

.venv/bin/python -m albench.celltype.search \
  --rounds 6 --per-round 3 \
  --epochs 8 --batch-size 512 --lr 0.005 \
  --subset 600000 --max-params 4000000 --max-sec 900 \
  --seed 0 --out outputs/celltype/cs42
