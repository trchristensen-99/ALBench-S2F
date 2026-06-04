#!/bin/bash
# Copy the Malinois / lentiMPRA dataset bundle into the koo lab shared storage.
#
# WHY THIS SCRIPT EXISTS: christen is NOT in the 'koo lab staff' group, so cannot
# write anywhere under /grid/koo. Run this from an account that DOES have koo
# write access. It reads the source files straight out of christen's repo, whose
# data files are world-readable (-rw-r--r--) with traversable parent dirs (o+rx),
# so any non-wsbs user can read them — no pre-staging copy is needed.
#
# USAGE:
#   ./copy_to_koo_shared.sh            # uses the default koo lentiMPRA path
#   ./copy_to_koo_shared.sh <koo-shared-datasets-dir>   # override the parent dir
#
# This is the Gosai et al. 2023 lentiMPRA (Malinois training data), a DIFFERENT
# dataset from the existing lentimpra/agarwal_2025/ — so it lands in a sibling
# subdir: <parent>/gosai_2023/.

set -euo pipefail

# Parent dir under koo shared storage; default is the known lentiMPRA folder.
DEST="${1:-/grid/koo/home/shared/data/lentimpra}"
SRC="/grid/wsbs/home_norepl/christen/ALBench-S2F"
HERE="$(cd "$(dirname "$0")" && pwd)"
BUNDLE="$DEST/gosai_2023"

if [ ! -d "$DEST" ]; then
  echo "ERROR: destination '$DEST' does not exist or is not writable to you." >&2
  exit 1
fi

mkdir -p "$BUNDLE/controls" "$BUNDLE/model"

echo "== 1) Core MPRA dataset (all 3 cell types) =="
cp -v "$SRC/data/k562/DATA-Table_S2__MPRA_dataset.txt" "$BUNDLE/"

echo "== 2) Controls & SNV pairs =="
cp -v "$SRC/data/agarwal_2025/k562_all_controls_200bp.tsv"             "$BUNDLE/controls/"
cp -v "$SRC/data/agarwal_2025/k562_shuffled_controls_200bp.tsv"        "$BUNDLE/controls/"
cp -v "$SRC/data/agarwal_2025/k562_dinucleotide_shuffled_controls.csv" "$BUNDLE/controls/"
cp -v "$SRC/data/k562/train_snv_pairs_clean.tsv"                       "$BUNDLE/controls/"

echo "== 3) Trained Malinois model =="
cp -v "$SRC/data/malinois_artifacts__20211113_021200__287348.tar.gz"  "$BUNDLE/model/"

echo "== 4) README =="
cp -v "$HERE/README.md" "$BUNDLE/"

# Make the bundle group-readable for the lab (best-effort).
chmod -R g+rX "$BUNDLE" 2>/dev/null || true

echo
echo "Done. Bundle written to: $BUNDLE"
du -sh "$BUNDLE"
echo "Contents:"
find "$BUNDLE" -type f -exec ls -lh {} \; | awk '{print "  "$5"\t"$9}'
