# Working in this repo (for LLM coding agents and new contributors)

Read this before changing anything. It is short on purpose.

## What this project is

We are choosing what sequences to put on a ~3.2M-oligo MPRA library. A **reservoir
strategy** decides how candidate sequences are generated; an **acquisition
strategy** decides which of them to select. We train small student models on
oracle-labelled sequences from each strategy and compare how well they generalise,
so the deliverable is a *training corpus recommendation*, not a model.

## The 60-second orientation

```bash
uv sync                       # environment
albench doctor                # which data assets you have, and how to get the rest
albench list                  # every strategy and every tunable parameter
albench list --kind acquisition   # just the acquisition methods
albench generate --strategy random --n 1000 --out /tmp/x.npz   # needs no data at all
```

`albench doctor` is the first thing to run and the first thing to check when
something fails. Missing data is by far the most common cause of a crash, and the
error tells you how to obtain each file.

## Where things live

| path | what it is | should you edit it? |
|---|---|---|
| `albench/registry.py` | the strategy table: name → factory, adapter, parameters | yes, to add a strategy |
| `albench/reservoir/` | the samplers themselves | yes |
| `albench/acquisition/` | acquisition methods | yes |
| `albench/paths.py` | where every external data asset lives | only to declare a new asset |
| `albench/cli.py` | the `albench` command | rarely |
| `experiments/`, `scripts/` | research code, accreted over a year | treat as legacy; do not copy its patterns |

`experiments/` and `scripts/` contain a lot of one-off analysis. It is kept for
reproducibility. **It is not the API and not a style model.** New shared code goes
in `albench/`.

## Acquisition methods, and why the controls are not optional

BADGE and BatchBALD both have failure modes that return a full, plausible batch while
actually selecting at random, with nothing in the output to say so:

- **BADGE**: the last-layer gradient under the model's own prediction as a
  pseudo-label is exactly zero for every candidate, so k-means++ runs on identical
  zero vectors. We use the expected gradient outer product instead, weighted by
  epistemic/aleatoric. `badge_epistemic_only` and `badge_kmeanspp_only` exist so a
  gain can be attributed to the fusion rather than to either half.
- **BatchBALD**: an M-member posterior has rank <= M-1, so a 384-batch from 10
  members is ~375 arbitrary picks. The implementation warns and records
  `n_informative_`.

If you add an acquisition method, add its controls at the same time.

## Adding a reservoir strategy

1. Write the sampler in `albench/reservoir/`, subclassing `ReservoirSampler`.
   Its `generate()` may take whatever arguments it needs.
2. Add one `register(Spec(...))` call in `albench/registry.py`. The `adapter`
   field is where you say how to call your `generate()` from a `Context`.
3. Declare every knob in `params` with a default and a one-line help string.
   If it is not in `params`, nobody can sweep it, and that is the whole point.
4. Declare any data files in `albench/paths.py` with a real `how_to_get`.

Nothing else needs to change. No if/elif to extend, no new YAML file per parameter
value — that was the old design and it is what made the codebase hard to extend.

## Training students: the menu of configurations

The student is not one network but an ensemble of up to 33 model-written configurations (architecture source + training recipe), trained on
whatever labelled set you have and stacked on the validation set. `albench.ensemble.menu` is the library for it (`Menu`, `train_member`,
`MenuEnsemble`, a `SequenceModel` for the AL loop), `albench menu list|export|train|stack|predict` the command line, and
[docs/MENU.md](docs/MENU.md) says how many configurations to train per dataset size (K = 21 / 11 / 12 at 10k / 30k / 100k) and why. Do not copy
`scripts/program_oct22/menu_retrain.py`: it is the experiment script, kept so old runs can be reproduced. Two guards that are not obvious: the
validation set is the oracle_v3-labelled `val_v0c` asset (never train on it), and the library refuses to train with AdamW when a configuration
asks for Muon and the package is missing (the loop would otherwise fall back silently).

## House rules that are not obvious

- **No absolute paths in `albench/`.** Everything resolves through
  `albench.paths.resolve()`. The one exception is the `fallbacks` tuple, which is
  explicitly for "this happens to exist on the CSHL cluster".
- **Verify with a measurement, not with "it runs".** Several bugs here were silent:
  a sampler that returned 34% duplicate sequences, a parameter that was declared and
  never read, a transition/transversion knob that produced half the ratio it was set
  to. All of them ran fine and produced plausible output. When you change a sampler,
  measure the property you changed.
- **Pools are nested.** Use `outputs/grid_pools/k562_nested/{strategy}_d{D}.npz` (canonical, seed 42) or
  `outputs/grid_pools/k562_reps/{strategy}_r{1,2,3}_d{D}.npz` (independent labelled draws, one per independent run; build with
  `scripts/slice_replicate_pools.py`). Every size inside a pool is a prefix of the next, so a size comparison varies only the
  amount of data. `k562_cumulative/` is legacy: its sizes are separate draws (and for genomic the sizes differ in label
  distribution), so it only reproduces old results; set `HP_REQUIRE_NESTED=1` to make the loader refuse it.
- **Chromosome holdouts are real.** chr7/chr13 are test, chr19/21/X are validation.
  Any strategy that touches genomic sequence must exclude them. Check this.
- **Reservoirs are generated at 1M+ scale.** A per-sequence Python loop that is fine
  at n=1,000 will not finish at n=1,000,000. Vectorise.
- **Parameters get stamped into output artefacts.** Do not remove that; we have been
  burnt by caches whose provenance was unknowable.

## Testing

```bash
pytest tests/ -q                    # unit tests
python scripts/test_samplers_fast.py   # sampler behaviour against real data (needs assets)
ruff format . && ruff check .       # pre-commit runs both
```

## Conventions

- Python ≥3.11, `from __future__ import annotations`, type hints on public functions.
- Docstrings say *why*, not *what* — the code already says what.
- stdlib imports, blank line, third-party (ruff enforces `I001`).
