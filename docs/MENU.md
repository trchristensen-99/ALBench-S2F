# The menu: which model configurations to train on a dataset, and the ensemble they form

This is the part of the study that turns "an AutoResearch search found good architectures" into something you can run on any
labelled set: a fixed list of **33 configurations** (a model-written PyTorch network plus the training recipe it was found with), the
order in which to train them, how many to train at 10k / 30k / 100k sequences, and the ensemble that combines them.

Code: `albench/ensemble/menu.py` (library), `albench/ensemble/menus/` (the data), `albench menu ...` (command line),
`tests/test_menu.py`. Evidence for every choice below is in the evidence ledger (the artifact the study is argued from); the numbers
quoted here are from its K-curve card.

## What a configuration is

`menu_v0.json` holds 33 entries. Each has `source` (a `class Net(nn.Module)` that is executed in a restricted namespace by
`albench.arch_search.validate.build_and_check`: no imports beyond torch/math, at most 40M parameters, CPU forward check) and
`train_config` (optimizer, learning rate, weight decay, batch size, epochs, schedule; 60 to 100 epochs, almost all Muon + OneCycle).
They are the 8 best-by-validation models of each of four AutoResearch runs (Claude Sonnet 5.5 x3, Sonnet 4.5 x1; 30k genomic pools; 10 rounds
x 2 proposals) plus the reference LegNet. The menu is **not specific to a reservoir or a size**: it is retrained on whatever data you have.
That was tested, not assumed: fresh searches on other reservoirs and at 100k / 300k tie the carried menu within about 0.005 ensemble r, and
the architecture source (genomic, random, mixed, five-reservoir searches) changes the result by at most 0.007.

## How many to train (the K curve)

`albench/ensemble/menus/per_size.json` holds the training-priority order and K for each size.

| training sequences | configurations to train (one seed each) | H100 time for the set | why |
|---|---|---|---|
| 10,000 | all 30 usable | about 2 hours | cheap, so everything is trained; the 8 best of a larger set matter most when data are small and validation is noisy |
| 30,000 | all 30 usable | about 3 hours | same |
| 100,000 | 20 | about 5 hours | smallest K inside the tolerances is 16; 25% margin |
| 300,000 and up | 12 (chosen by the previous size's validation r) | | reduced-menu analysis: within 0.001 of the whole menu at 300k |

K is not a smooth function of size because it is the smallest number that stays inside a tolerance of a noisy reference (the 8 best of all 30 by
validation r), measured on three reservoirs with one to three pools each. At 10k the raw threshold was 17 to 30 depending on the evaluation set, at 30k 11 to 12, at 100k 9 to 16. The table above
therefore applies a cost rule instead: train everything when the whole set costs under about 3 H100-hours (10k, 30k), otherwise the smallest K within
0.002 (genomic test) or 0.003 (designed set, random sequences) plus a 25% margin (100k), and a smaller reduced set from 300k, where each model costs hours. The loss against all 30 at the
8-member mean (genomic test / designed set), mean over reservoirs: 10k K = 12: 0.005 / 0.008, K = 20: 0.002 / 0.005; 30k K = 12: 0.002 / 0.003, K = 20: 0.0003 / 0.001; 100k K = 12: 0.001 / 0.006, K = 20: 0.000 / 0.000.
The study's standard ensemble is not the 8-member mean but the non-negative stack over everything trained, which is a further 0.001 to 0.003 higher, so the stack over 30 configurations
is the number to aim for where it is affordable. K counts the 30 usable configurations (three of the 33 are left out, see below); the 10k row and the random / motif columns at 10k and 100k rest
on one pool each until the replicate pools finish.

How the curve was made (figure `kcurve.png` in the ledger and in the shared folder): the menu was retrained on every size x reservoir cell
(genomic, random, motif; three replicate pools where they exist); for each K the first K configurations in a given order are taken, the 8 with
the best retrained validation r are averaged, and the test r is compared with the same 8-member ensemble made from all 33.

* **Order matters, a little.** Taking the best two models of every search run first and then everything by search validation r reaches the
  tolerance (genomic test) with 3 to 6 fewer configurations than a random order (random order: 20, 17 and 13 at 10k, 30k and 100k, against 17, 11 and 9). That
  is the rule stored in `per_size.json` (`rule: first2`). Ranking by search validation r alone is about as good at 30k and 100k and worse at 10k.
* **More configurations beat more seeds.** With three seeds per configuration the K needed is not lower at 10k (about 16) or 100k (12 or more);
  at 30k six configurations x three seeds (18 models) reach the tolerance, which is more training than 11 configurations x one seed.
  (Pool r1 only for the seed curves.)
* **Weights beat a mean of 8.** The non-negative stack over all K trained models (the study's standard ensemble) is 0.001 to 0.003 above the
  8-member mean at every K; if you ensemble by weights rather than a fixed 8-member mean, one or two fewer configurations are enough.
* The designed set (out-of-distribution) needs a few more configurations than the genomic test set at 100k (12 vs 9).

`MenuEnsemble.for_size(size, ...)` and `albench menu list --size N` use this table; `Menu.for_size(size, k=..., rule=...)` overrides it.

## Using it

Command line (all of it is resumable; `--shard i/n` splits a size across Slurm jobs):

```bash
albench menu list --size 30000                       # configurations in priority order, K / seeds / patience for the size
albench menu export --size 30000 --out cfg30k.json   # just the configurations, as a menu.json for your own trainer
albench menu train --size 30000 --pool pool.npz --out runs/30k --shard 0/4      # one of four jobs
albench menu stack --out runs/30k                    # after all shards: fit the stack, write ensemble.json
albench menu predict --ensemble runs/30k --sequences cand.npz --out preds.npz   # ensemble + per-member predictions
```

`--pool` is an `.npz` with `sequences`, `oracle_labels` and `oracle_id` (= `oracle_v3`; an unstamped set is refused unless you pass
`--allow-unstamped`). The validation set defaults to the `val_v0c` asset.

Python, as the student of an active-learning loop (`albench.core.loop.ALLoop` calls `fit` every round and `save` after it):

```python
from albench.ensemble import MenuEnsemble, load_menu
from albench.ensemble.menu import load_labelled
from albench.paths import resolve

val_s, val_y = load_labelled(resolve("val_v0c"))            # chr19/21/X sequences labelled by oracle_v3 ("V0c")
student = MenuEnsemble.for_size(30_000, val_s, val_y, devices=["cuda:0", "cuda:1"])   # K, seeds, patience from the table
student.fit(train_seqs, train_labels)                       # trains every configuration, then fits the stack
student.predict(seqs); student.member_predictions(seqs)     # (members, N) for disagreement-based acquisition
student.epistemic_uncertainty(seqs)                         # std over the members the stack uses
```

Acquisition: `EnsembleAcquisition` works on a `MenuEnsemble` unchanged (it reads `student.models` / `_predict_member`);
`uncertainty` / `epistemic_uncertainty` is the standard deviation across members with positive stack weight; `embed` raises
`NotImplementedError` (the members are arbitrary architectures with no common embedding), so embedding-based methods must take their
documented fallback. `fit` skips retraining when it is handed the training set the ensemble was loaded with, so the first round of a loop
that starts from a saved ensemble does not retrain it.

### The saved genomic ensembles (10k and 30k starting points)

`menu_ensembles` asset (CSHL: `/grid/koo/home/shared/albench/ensembles`): for 10,000 and 30,000 genomic sequences (the first
10k / 30k of `outputs/grid_pools/k562_nested/genomic_d{D}.npz`, seed 42, labelled by oracle_v3), all 33 configurations x 3 seeds trained
with the recipe below, with weights, per-member validation predictions and the stack. Load one with
`MenuEnsemble.load(dir, val_s, val_y)`; pick the K best of them with `ensemble.json` if you want the recommended subset (members are listed
with their weights and validation r). Before this, only predictions had been stored, never weights.

## Things that will bite

* **The validation set is the standard `V0c`** (29,449 real chr19/21/X sequences, oracle_v3 labels). It drives early stopping *and* the stack
  weights, so its r is optimistic; report test sets only. Never put those sequences in a training set. Results from the previous validation
  set (V0, other-oracle labels) are usable: V0c moved the genomic-test ensemble by +0.002 on average, with the same reservoir ordering.
* **Muon.** Most configurations use the Muon optimizer. The training loop silently falls back to AdamW if the `muon` package is missing, which
  trains different models; the library refuses unless `ALBENCH_ALLOW_OPTIMIZER_FALLBACK=1`. The repo `.venv` on the cluster has it vendored.
* **Patience.** The recipe is early stopping with patience 16 (the standard since Oct 28) and best-weight restore, reverse-complement averaged MSE, shift augmentation up
  to 10 bp, gradient clipping 1.0, fp32. Patience 16 beat 8 in every cell of the first recipe study (+0.001 at 100k to +0.011 at 10k) for 1.0 to 1.6x
  the training time; patience 24 and 32 are being tested, and replicate pools for 16 against 8 are running. Results in the lab ledger from before Oct 28 used patience 8, and the
  ensembles in `ensembles/` are labelled with the patience they were trained with.
* **Three of the 33 configurations are left out** (`per_size.json["skip"]`): two exceed the 8 s CPU forward-time check on 10 to 14% of
  retrainings (node dependent; they were missing from some pools and not others), and one produces predictions of order 1e5 on
  distribution-shifted sets in about 17% of retrainings. The shipped recipe has 30 usable configurations, a fixed priority order and no timing
  gate, so every shard of a sharded run agrees on the list. Every member is also screened on random, GC-extreme and repetitive sequences before
  it can enter an ensemble (`probe_sequences`); a member that explodes is left out, and at prediction time a member that explodes on a batch
  is masked for that batch with a warning.
* **Cost at 100k.** About 12 H100-minutes per model, so K = 12 is a few GPU-hours per training set; use `--shard` rather than a single process,
  and `MenuEnsemble(devices=[...])` for one worker per GPU. `MenuEnsemble.fit` retrains from scratch every round, as the loop's other students do.
* **Reproducing an experiment run.** `train_member(..., index=<position in menu_v0.json>, seed=s)` seeds torch and numpy exactly as
  `scripts/program_oct22/menu_retrain.py` did (1000 + 100 * seed + index), so a library retrain matches a stored run up to GPU non-determinism.

## Where things live

| what | where |
|---|---|
| configurations, priority order, K per size | `albench/ensemble/menus/menu_v0.json`, `per_size.json` (regenerate with `scripts/analysis/make_per_size.py`) |
| K-curve analysis | `scripts/analysis/analyze_oct27.py --only kcurve`, `figs_oct28.py` |
| library, command line | `albench/ensemble/menu.py`, `albench/ensemble/stack.py` (`fit_zscored_stack`), `albench menu` |
| shared lab folder (CSHL) | `/grid/koo/home/shared/albench/` (`README.md`, `menu/`, `ensembles/`, `data/` with V0c, test sets and the genomic start sets, `oracle_v3_s2/`, and a code snapshot `ALBench-S2F/`) |
| the legacy experiment scripts | `scripts/program_oct22/menu_retrain.py` (+ `build_menu.py`): kept for reproducibility, not a style model |
