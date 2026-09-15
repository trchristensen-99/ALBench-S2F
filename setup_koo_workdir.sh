#!/bin/bash
# Stand up a working copy on the KOO fileset (20 TB free) so NEW large outputs stop consuming the
# shared WSBS student fileset (10 TB, 100% full, 93 users). The WSBS copy is left untouched.
#   code: shallow clone (small .git) + rsync of the live working tree's scripts/launchers
#   data: only the inputs needed to run (battery + caches), ~3.5 GB
#   venv: symlinked back to WSBS (15 GB) instead of duplicated
set -euo pipefail
SRC=/grid/wsbs/home_norepl/christen/ALBench-S2F
DST=/grid/koo/home/christen/ALBench-S2F

echo "=== 1. shallow clone code -> $DST ==="
if [ ! -d "$DST/.git" ]; then
  git clone --depth 1 "file://$SRC" "$DST" 2>&1 | tail -3
  git -C "$DST" remote set-url origin https://github.com/trchristensen-99/ALBench-S2F.git
else
  echo "  already cloned"
fi

echo "=== 2. sync live scripts/launchers (picks up uncommitted + scp'd files) ==="
rsync -a "$SRC/scripts/" "$DST/scripts/"
rsync -a "$SRC/albench/" "$DST/albench/" 2>/dev/null || true
rsync -a "$SRC/experiments/" "$DST/experiments/" 2>/dev/null || true
for f in "$SRC"/*.sh "$SRC"/*.sbatch "$SRC"/pyproject.toml "$SRC"/uv.lock; do
  [ -f "$f" ] && cp -p "$f" "$DST/" 2>/dev/null || true
done

echo "=== 3. copy minimal run inputs (~3.5 GB) ==="
mkdir -p "$DST/outputs" "$DST/data/k562" "$DST/logs"
rsync -a "$SRC/outputs/chr_split_cache/"        "$DST/outputs/chr_split_cache/"
rsync -a "$SRC/data/k562/test_sets_ag_s2_chrsplit/" "$DST/data/k562/test_sets_ag_s2_chrsplit/"
rsync -a "$SRC/outputs/reservoir_cache/"        "$DST/outputs/reservoir_cache/"
rsync -a "$SRC/outputs/replay_anchors/"         "$DST/outputs/replay_anchors/" 2>/dev/null || true

echo "=== 4. venv: symlink to WSBS (avoid duplicating 15 GB) ==="
[ -e "$DST/.venv" ] || ln -s "$SRC/.venv" "$DST/.venv"

echo "=== 5. verify ==="
du -sh "$DST" 2>/dev/null
ls "$DST/outputs/chr_split_cache" | head -2
echo "  battery sets: $(ls "$DST/data/k562/test_sets_ag_s2_chrsplit"/*_oracle.npz 2>/dev/null | wc -l)"
echo "  reservoir caches: $(ls "$DST/outputs/reservoir_cache"/*.npz 2>/dev/null | wc -l)"
echo "SETUP_DONE"
