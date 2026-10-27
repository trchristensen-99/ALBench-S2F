#!/bin/bash
# Re-run every ledger analysis and figure script on the cluster (CPU only, about 10 minutes). Writes outputs/justification_oct22/refresh.done when finished.
#   /cm/shared/apps/slurm/current/bin/sbatch --partition=cpuq --qos=cpuq_base --cpus-per-task=8 --mem=24G --time=03:00:00 --export=ALL,REFRESH_THREADS=8 \
#       --output=outputs/justification_oct22/refresh.log --wrap="bash scripts/program_oct22/refresh_analyses.sh"
cd "$(dirname "$0")/../.." || exit 1
rm -f outputs/justification_oct22/refresh.done
# Heavy numpy work must not run on the login/development node (it mails the user an "excessive usage" warning at about 1000% average core usage):
# submit this script to the cpuq partition (refresh_ledger.sh does) or, if it has to run on the login node, leave REFRESH_THREADS at 2.
THREADS=${REFRESH_THREADS:-2}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
PY=".venv/bin/python"
$PY scripts/analysis/analyze_oct26.py --only noise,acq,acq3,grid,agent2,agent_grid,zoo,hg,seeds,ladder,anchor,g300,live
$PY scripts/analysis/analyze_oct27.py --only coord,extrap,reduced,ensmethods,recipe,valdep,hgdep,loopctl,hptune,unseen_menu,kcurve,recipe2,plain_grid
$PY scripts/analysis/figs_oct26.py
$PY scripts/analysis/figs_oct27.py
$PY scripts/analysis/figs_oct28.py
date > outputs/justification_oct22/refresh.done
