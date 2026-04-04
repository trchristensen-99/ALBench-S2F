#!/bin/bash
# Backup all result.json, predictions.npz, and key config files.
# Creates a timestamped backup directory.
#
# Run from login node (no GPU needed):
#   bash scripts/slurm/backup_results.sh
#
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="outputs/results_backup_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

echo "=== Backing up results to ${BACKUP_DIR} ==="
echo "Started: $(date)"

# Copy all result.json files preserving directory structure
find outputs/ -name "result.json" -not -path "*/results_backup*" | while read -r f; do
    dest="${BACKUP_DIR}/${f}"
    mkdir -p "$(dirname "$dest")"
    cp "$f" "$dest"
done
echo "  result.json files: $(find "$BACKUP_DIR" -name "result.json" | wc -l)"

# Copy all predictions.npz files
find outputs/ -name "predictions.npz" -not -path "*/results_backup*" | while read -r f; do
    dest="${BACKUP_DIR}/${f}"
    mkdir -p "$(dirname "$dest")"
    cp "$f" "$dest"
done
echo "  predictions.npz files: $(find "$BACKUP_DIR" -name "predictions.npz" | wc -l)"

# Copy all training_history.json files
find outputs/ -name "training_history.json" -not -path "*/results_backup*" | while read -r f; do
    dest="${BACKUP_DIR}/${f}"
    mkdir -p "$(dirname "$dest")"
    cp "$f" "$dest"
done

# Copy pretrained eval
if [ -d "outputs/malinois_pretrained_eval" ]; then
    cp -r outputs/malinois_pretrained_eval "$BACKUP_DIR/outputs/malinois_pretrained_eval"
fi
if [ -d "outputs/malinois_pretrained_eval_v2" ]; then
    cp -r outputs/malinois_pretrained_eval_v2 "$BACKUP_DIR/outputs/malinois_pretrained_eval_v2"
fi

# Size
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo ""
echo "Backup complete: ${BACKUP_DIR} (${BACKUP_SIZE})"
echo "Finished: $(date)"
