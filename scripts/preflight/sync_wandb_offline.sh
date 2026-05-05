#!/bin/bash
# Sync all offline W&B runs in wandb/offline-run-* to the cloud project.
# Run this from the project root after WANDB_API_KEY is set + wandb logged in.
#
# Each offline run takes ~5-15 sec to sync. With ~100s of runs, total
# sync time is a few minutes.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

if [ ! -d wandb ]; then
    echo "No wandb/ directory — nothing to sync"
    exit 0
fi

# Sync everything under wandb/offline-run-* in parallel batches of 4.
mapfile -t RUNS < <(find wandb -maxdepth 1 -name 'offline-run-*' -type d 2>/dev/null | sort)
N=${#RUNS[@]}
if [ "$N" -eq 0 ]; then
    echo "No offline runs found"
    exit 0
fi
echo "Syncing $N offline runs to W&B cloud (entity=trchristensen-99, project=albench-s2f) …"

for ((i=0; i<N; i++)); do
    run="${RUNS[$i]}"
    echo "  [$i/$N] syncing $(basename $run)"
    uv run --no-sync wandb sync "$run" 2>&1 | tail -1 || echo "  WARN: sync failed for $run"
done

echo "Done. Visit https://wandb.ai/trchristensen-99/albench-s2f to view runs."
