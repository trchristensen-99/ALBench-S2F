#!/bin/bash
# Re-run every ledger analysis and figure script on the cluster (CPU only, about 10 minutes). Writes outputs/justification_oct22/refresh.done when finished.
#   nohup setsid bash scripts/program_oct22/refresh_analyses.sh > outputs/justification_oct22/refresh.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
rm -f outputs/justification_oct22/refresh.done
PY=.venv/bin/python
$PY scripts/analysis/analyze_oct26.py --only noise,acq,acq3,grid,agent2,agent_grid,zoo,hg,seeds,ladder,anchor,g300,live
$PY scripts/analysis/analyze_oct27.py --only coord,extrap,reduced,ensmethods,recipe,valdep,hgdep,loopctl,hptune
$PY scripts/analysis/figs_oct26.py
$PY scripts/analysis/figs_oct27.py
date > outputs/justification_oct22/refresh.done
