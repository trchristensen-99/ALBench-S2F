# Environment

Two levels, so that nobody installs the heavy one by accident:

| you want to | install | files |
|---|---|---|
| train / stack / load / evaluate menu ensembles, write acquisition code, run the tests | PyTorch + numpy + scipy + scikit-learn + the Muon optimizer | `requirements/menu.txt` |
| also label newly acquired sequences with the oracle | JAX + alphagenome + alphagenome_ft (+ the AlphaGenome backbone weights and the oracle checkpoints) | `requirements/oracle.txt` |

```bash
scripts/setup_env.sh              # first level; add --oracle for the second
source .venv/bin/activate && export PYTHONPATH=$PWD:$PWD/experiments
python -m albench.cli doctor      # python packages, GPU, Muon, and which data assets were found
```

Why a recipe and not a ready-made shared environment: a virtual environment is tied to the absolute paths and interpreter it was built with, so a shared copy breaks as
soon as it is moved or the cluster's Python module changes; anything installed into a shared environment by one person changes it for everybody; and the PyTorch / JAX CUDA
builds should match the node you run on. `requirements/lock_full.txt` is the complete frozen list (240 packages) of the environment the study ran in, for reference when something
needs to be reproduced exactly; the two short files are what you should install.

Things the environment must have that `pip install` of the repository does not give you:

* **Muon** (`muon.py`, MIT licence, Keller Jordan): most of the model configurations use this optimizer and it is not on PyPI. `scripts/setup_env.sh` copies the lab's vendored file from
  `/grid/koo/home/shared/albench/vendor/muon.py` (or installs it from GitHub when that is not readable). The training loop silently falls back to AdamW without it; the library refuses to.
* **alphagenome_ft** (MIT licence, Alan Murphy; finetuning utilities for AlphaGenome), source in `/grid/koo/home/shared/albench/vendor/alphagenome_ft`, installed by `setup_env.sh --oracle`.
* **AlphaGenome backbone weights** (`ag_weights` asset): `/grid/koo/home/shared/albench/alphagenome_weights/alphagenome-jax-all_folds-v1`, found automatically by `albench doctor`.
* **The oracle** (`oracle` asset): `/grid/koo/home/shared/albench/oracle_v3_s2`; set both `ALBENCH_ORACLE_DIR` and `AG_S2_ORACLE_DIR` to it if you keep a copy elsewhere.
* A GPU for training (an H100 or V100 node). Listing, exporting, loading and predicting with a small ensemble work on CPU.

Known version sensitivity: scikit-learn 1.9 removed `ElasticNetCV(n_alphas=...)`; the stack code handles both, but pin `scikit-learn==1.8.0` as in the requirements to reproduce our numbers.
