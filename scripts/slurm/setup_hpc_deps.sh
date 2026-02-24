#!/bin/bash
# Install HPC-specific AlphaGenome packages into the uv venv after uv sync.
# Source this file from Slurm scripts before calling uv run python.
# Packages not on PyPI that must come from HPC-local paths or GitHub:
#   - alphagenome     (PyPI: base DeepMind SDK, not declared as dep by alphagenome_ft)
#   - alphagenome_ft   (local: ~/alphagenome_ft-main)
#   - alphagenome_research  (GitHub: google-deepmind/alphagenome_research)
#   - jmp              (PyPI: mixed-precision for JAX)
#   - jaxtyping        (PyPI: type annotations for JAX)
#   - dm-haiku         (PyPI: Haiku neural network library)
#   - chex             (PyPI: JAX test utilities)
#   - orbax-checkpoint (PyPI: JAX checkpoint library)

AG_FT_PATH="${HOME}/alphagenome_ft-main"
AG_RES_REV="35ea7aa5"

_check() { uv run python -c "import $1" 2>/dev/null; }

if ! _check alphagenome; then
  echo "[setup_hpc_deps] Installing alphagenome ..."
  uv pip install "alphagenome==0.6.0" || echo "[setup_hpc_deps] WARNING: alphagenome install failed"
fi

# aiohttp and requests are declared deps of alphagenome_ft but not installed
# because alphagenome_ft is installed with --no-deps. Install them explicitly.
if ! _check aiohttp; then
  echo "[setup_hpc_deps] Installing aiohttp ..."
  uv pip install "aiohttp" || echo "[setup_hpc_deps] WARNING: aiohttp install failed"
fi

if ! _check requests; then
  echo "[setup_hpc_deps] Installing requests ..."
  uv pip install "requests" || echo "[setup_hpc_deps] WARNING: requests install failed"
fi

if ! _check alphagenome_ft; then
  if [ -d "$AG_FT_PATH" ]; then
    echo "[setup_hpc_deps] Installing alphagenome_ft from $AG_FT_PATH ..."
    uv pip install "$AG_FT_PATH" || echo "[setup_hpc_deps] WARNING: alphagenome_ft install failed"
  else
    echo "[setup_hpc_deps] WARNING: $AG_FT_PATH not found; skipping alphagenome_ft"
  fi
fi

if ! _check alphagenome_research; then
  echo "[setup_hpc_deps] Installing alphagenome_research from GitHub (with deps) ..."
  uv pip install \
    "alphagenome-research @ git+https://github.com/google-deepmind/alphagenome_research@${AG_RES_REV}" \
    || echo "[setup_hpc_deps] WARNING: alphagenome_research install failed"
fi

if ! _check jmp; then
  echo "[setup_hpc_deps] Installing jmp ..."
  uv pip install "jmp==0.0.4" || echo "[setup_hpc_deps] WARNING: jmp install failed"
fi

if ! _check jaxtyping; then
  echo "[setup_hpc_deps] Installing jaxtyping ..."
  uv pip install "jaxtyping" || echo "[setup_hpc_deps] WARNING: jaxtyping install failed"
fi

if ! _check haiku; then
  echo "[setup_hpc_deps] Installing dm-haiku ..."
  uv pip install "dm-haiku" || echo "[setup_hpc_deps] WARNING: dm-haiku install failed"
fi

if ! _check chex; then
  echo "[setup_hpc_deps] Installing chex ..."
  uv pip install "chex" || echo "[setup_hpc_deps] WARNING: chex install failed"
fi

if ! _check orbax.checkpoint; then
  echo "[setup_hpc_deps] Installing orbax-checkpoint ..."
  uv pip install "orbax-checkpoint" || echo "[setup_hpc_deps] WARNING: orbax-checkpoint install failed"
fi
