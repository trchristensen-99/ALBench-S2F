#!/bin/bash
# Verify HPC-specific AlphaGenome packages are present in the uv venv.
# Source this file from Slurm scripts BEFORE calling uv run --no-sync python.
#
# This script is CHECK-ONLY: it never modifies the venv.
# All packages must be pre-installed interactively from the login node.
# Run scripts/install_hpc_packages.sh once from the login node to set things up.

# ── Disk space reservation ────────────────────────────────────────────────────
# Release the placeholder file at job start to free space for outputs.
# Recreate it on job exit if no other user jobs are still running.
_PLACEHOLDER="/grid/wsbs/home_norepl/christen/.disk_placeholder.dat"
_PLACEHOLDER_SIZE="20G"

if [ -f "$_PLACEHOLDER" ]; then
    rm -f "$_PLACEHOLDER"
    echo "[setup_hpc_deps] Released ${_PLACEHOLDER_SIZE} disk reservation"
fi

_recreate_placeholder() {
    # Only recreate if no other SLURM jobs are running/pending for this user
    local active
    active=$(/cm/shared/apps/slurm/current/bin/squeue -u "$(whoami)" -h -t R,PD 2>/dev/null | wc -l)
    if [ "${active:-0}" -le 1 ]; then
        fallocate -l "$_PLACEHOLDER_SIZE" "$_PLACEHOLDER" 2>/dev/null && \
            echo "[setup_hpc_deps] Recreated ${_PLACEHOLDER_SIZE} disk reservation" || true
    fi
}
trap _recreate_placeholder EXIT

# ── Package checks ────────────────────────────────────────────────────────────
_check() { .venv/bin/python -c "import $1" 2>/dev/null; }

_MISSING=0

if ! _check alphagenome; then
  echo "[setup_hpc_deps] WARNING: alphagenome not found — run scripts/install_hpc_packages.sh"
  _MISSING=1
fi
if ! _check alphagenome_ft; then
  echo "[setup_hpc_deps] WARNING: alphagenome_ft not found — run scripts/install_hpc_packages.sh"
  _MISSING=1
fi
if ! _check alphagenome_research; then
  echo "[setup_hpc_deps] WARNING: alphagenome_research not found — run scripts/install_hpc_packages.sh"
  _MISSING=1
fi
if ! _check jmp; then
  echo "[setup_hpc_deps] WARNING: jmp not found — run scripts/install_hpc_packages.sh"
  _MISSING=1
fi
if ! _check haiku; then
  echo "[setup_hpc_deps] WARNING: dm-haiku not found — run scripts/install_hpc_packages.sh"
  _MISSING=1
fi
if ! _check orbax.checkpoint; then
  echo "[setup_hpc_deps] WARNING: orbax-checkpoint not found — run scripts/install_hpc_packages.sh"
  _MISSING=1
fi

if [ "$_MISSING" -ne 0 ]; then
  echo "[setup_hpc_deps] ERROR: required packages are missing. Aborting job."
  exit 1
fi

echo "[setup_hpc_deps] All packages OK."
