#!/bin/bash
# Re-run every ledger analysis and figure script on the cluster (CPU only, about 10 minutes). Writes outputs/justification_oct22/refresh.done when finished.
#   nohup setsid bash scripts/program_oct22/refresh_analyses.sh > outputs/justification_oct22/refresh.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
rm -f outputs/justification_oct22/refresh.done
# The login/development node flags jobs that average many cores: numpy/scipy start one BLAS thread per core, so cap them (the analyses take longer but stay at about two cores)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY="nice -n 10 .venv/bin/python"
$PY scripts/analysis/analyze_oct26.py --only noise,acq,acq3,grid,agent2,agent_grid,zoo,hg,seeds,ladder,anchor,g300,live
$PY scripts/analysis/analyze_oct27.py --only coord,extrap,reduced,ensmethods,recipe,valdep,hgdep,loopctl,hptune,unseen_menu
$PY scripts/analysis/figs_oct26.py
$PY scripts/analysis/figs_oct27.py
date > outputs/justification_oct22/refresh.done
