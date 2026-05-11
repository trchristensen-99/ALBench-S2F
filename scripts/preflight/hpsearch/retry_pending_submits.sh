#!/bin/bash
# Retry the AutoResearch submissions that hit QOSMaxSubmitJobPerUserLimit.
# Run periodically to catch slots as they free up.
#
# What we still need to submit (after first round):
#   - DREAM-RNN D=5k: roles B, C  (A was submitted)
#   - DREAM-ATTN D=5k: roles A, B, C
#   - LegNet D=100k: roles A, B, C  (these need ≥6h, won't fit fast qos)
#   - DREAM-RNN D=100k: roles A, B, C
#   - DREAM-ATTN D=100k: roles A, B, C

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
source .venv/bin/activate

# D=5k cells → fast qos (short jobs)
for arch in dream_rnn dream_attn; do
    python -m scripts.preflight.hpsearch.autoresearch_orchestrator \
        submit-round --arch "$arch" --d_train 5000 --round_idx 0 --qos fast 2>&1 \
        | grep -E '→|sbatch' | head -5
done

# D=100k cells → default qos (need 6h)
for arch in legnet dream_rnn dream_attn; do
    python -m scripts.preflight.hpsearch.autoresearch_orchestrator \
        submit-round --arch "$arch" --d_train 100000 --round_idx 0 --qos default 2>&1 \
        | grep -E '→|sbatch' | head -5
done
