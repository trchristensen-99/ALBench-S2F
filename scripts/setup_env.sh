#!/bin/bash
# Build a Python environment for the shared workflow.
#   scripts/setup_env.sh            # menu training / evaluation / acquisition (CPU or GPU torch)
#   scripts/setup_env.sh --oracle   # plus JAX, alphagenome and alphagenome_ft for labelling sequences with the oracle
# Environment variables: ENV_DIR (default .venv), MUON_PY (path to muon.py; default the lab shared copy), ALPHAGENOME_FT (path to the alphagenome_ft source).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_DIR=${ENV_DIR:-.venv}
SHARED=/grid/koo/home/shared/albench
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
uv venv "$ENV_DIR" --python 3.11
uv pip install --python "$ENV_DIR/bin/python" -r requirements/menu.txt
SITE=$("$ENV_DIR/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
if [ -r "${MUON_PY:-$SHARED/vendor/muon.py}" ]; then
  cp "${MUON_PY:-$SHARED/vendor/muon.py}" "$SITE/muon.py"
else
  uv pip install --python "$ENV_DIR/bin/python" "git+https://github.com/KellerJordan/Muon"
fi
if [ "${1:-}" = "--oracle" ]; then
  uv pip install --python "$ENV_DIR/bin/python" -r requirements/oracle.txt
  uv pip install --python "$ENV_DIR/bin/python" "${ALPHAGENOME_FT:-$SHARED/vendor/alphagenome_ft}"
fi
echo
echo "Done. Use it with:   source $ENV_DIR/bin/activate && export PYTHONPATH=\$PWD:\$PWD/experiments"
echo "Check it with:       python -m albench.cli doctor"
