# Benchmark: genomic ensembles against MPRA-LegNet and AlphaGenome-FT-MPRA (Oct 28)

Everything below is measured on the same training pools (nested canonical genomic pool, 30k and 10k, oracle_v3 labels) and scored against oracle_v3 on the same
test sets. Figures: `bench_genomic_ensembles.png` (all arms, three test sets, 30k and 10k), `bench_zoom_without_alphagenome.png` (the same without AlphaGenome so the
menu-LegNet gap is readable), `bench_ensemble_growth.png` (ensemble r against the number of members). The numbers are in `data_bench33.json`; the table at the end of this file is
generated from it.

## What the benchmark says

- **AlphaGenome-FT-MPRA is far ahead of everything else, at both sizes.** At 30k one Stage 2 model reaches r = 0.961 on the genomic test set (0.803 on the designed set,
  0.970 on random sequences), against 0.745 for one menu architecture, 0.757 for one tuned MPRA-LegNet and 0.728 for one published-recipe MPRA-LegNet. At 10k it is 0.950 against
  0.651, 0.642 and 0.611. Its seed-to-seed spread is below 0.001, so ensembling adds little (6 models: 0.963 at 30k; 5 models: 0.951 at 10k), whereas ensembling 20 LegNets adds
  +0.025 (30k) and +0.037 (10k).
- **Stage 2 from Stage 1 beats Stage 1 alone** by +0.023 (30k) and +0.022 (10k) on the genomic test set and +0.026 / +0.027 on the designed set.
- **Our 20-architecture menu is not better than 20 well-tuned MPRA-LegNets at 30k, and is better at 10k.** Stacked on V0c, genomic test r at 30k: menu 0.782, LegNet b256 0.782,
  LegNet e100 0.781, LegNet with the published schedule 0.750. At 10k: menu 0.700, LegNet b256 0.679, e100 0.664, published 0.647. On the designed set the menu leads at 10k
  (0.180 against 0.163) and is slightly behind at 30k (0.271 against 0.279); on random sequences the menu is level or ahead (0.711 against 0.713 at 30k, 0.595 against 0.550 at 10k).
  The published schedule (batch 1024, 25 epochs, made for ~600k sequences) is clearly too short at 10k to 30k; most of what the menu seems to add over "the published LegNet" at 30k is that schedule.
- **Against the experimental labels** (genomic test set, oracle versus experiment is 0.972): AlphaGenome-FT S2 0.916 (single) and 0.917 (6 models) at 30k, 0.905 and 0.906 at
  10k; menu 0.742 and LegNet x20 0.742 at 30k.

## Read this before using the numbers

1. **oracle_v3 is itself a 10-member AlphaGenome-encoder ensemble.** An AlphaGenome student trained on its labels distils its own family: the table measures how well each
   architecture can be fitted to the oracle from D labelled sequences, not independent ground truth. The experimental-label column is the part that is not circular, and the gap
   there (0.917 against 0.742 at 30k) is much smaller than the gap against the oracle (0.963 against 0.782), though it is bounded above by 0.972.
2. **The menu retrains validated on a different labelling of the V0c sequences.** `sc31_*` used `outputs/chr_split_cache/chr_val_ref_only.npz` (the same 29,449 sequences as
   V0c in the same order, but oracle labels that differ by r = 0.983, max difference 3.3), not `val_v0c`. Every arm here is stacked on V0c; the menu's stored validation
   predictions were re-used with the V0c labels. Early stopping inside the menu models was on the other labelling.
3. **Hardware and precision.** AlphaGenome models trained on H100 (Stage 2 needs about 55 GB); LegNets on V100 or H100 with the same fp16 autocast code path.
4. **LegNet schedules were compared on V0c validation** (three fixed schedules x 20 seeds, no further search); `b256` (batch 256, 60 epochs) has the best mean validation r at both
   sizes and is the "tuned" arm. The published-recipe arm is kept in every table.
5. **AlphaGenome recipes.** The ensemble rows use the `oracle` Stage 2 recipe (the one that trained the oracle members): 6 seeds at 30k and 5 at 10k at the time of this analysis; the
   `winner` recipe (memory `project_ag_s2_winner`: cosine decay and 3 warm-up epochs) has one seed per size and is +0.001 to +0.002 above the oracle recipe on the genomic test set (one seed per size; the seed sd is 0.0005 to 0.001),
   so it is reported but not used for the ensembles. More oracle-recipe seeds (30k s6 to s9, 10k s5 to s9) were still queued; the analysis is a pure function of the
   model directories and can be re-run (command below) to add them. A first 30k probe with a 20-epoch cap (still improving) was set aside in `outputs/bench33/probe_20ep`; the shipped code has a 60-epoch cap.
6. **Seeds** are fresh (`os.urandom`, `seeds.json`), not pinned; subsets of 5 and 10 members are random subsets of the available seeds (8 draws).
7. Stage 2 starts from Stage 1 in the same process through a checkpoint written to disk and loaded back; each job asserts that the loaded model reproduces the Stage 1
   validation Pearson before any Stage 2 step (`s1_loaded`, `s2_start_val_after_loading_s1` in every meta file; the difference is 0.0 in all 13 models trained so far).

## Re-running

```
# on the cluster, cpuq: reads outputs/menu_retrain_v0/{sc31_*_genomic,bench33_*}, writes outputs/justification_oct22/data_bench33.json
python scripts/analysis/analyze_bench33.py
# anywhere, with data_bench33.json in the folder:
python scripts/analysis/make_bench33_figures.py --in slide_figures/benchmarks --out slide_figures/benchmarks
```
Cells: `scripts/program_oct22/register_bench33.py` (LegNet cells `bench33_ln_*`, AlphaGenome cells `bench33_ag_*`); trainers `bench33_legnet.py`, `bench33_ag.py`, shared code `bench33_common.py`.


## Training details of every arm

All arms train on the nested canonical genomic pool of the size (`outputs/grid_pools/k562_nested/genomic_d{D}.npz`, replicate 0, oracle_v3 labels), are
early-stopped / checkpoint-selected on V0c (`outputs/val_variants/val_V0.npz`, oracle_v3 labels, 29,449 chr19/21/X sequences; never trained on) and are
scored against oracle_v3 on the 18 test sets of `scaling_hp_search.load_all_test_sets()` (the same ones as the menu retrains). Predictions are reverse-complement
averaged; forward-only predictions are stored too (`fwd_*` in each npz). Training seeds were drawn once from `os.urandom` (`seeds.json` in this folder) and are
recorded in every meta file; none is pinned. The ensemble "stack" is the positive ElasticNetCV of `albench.ensemble.stack.fit_zscored_stack` fitted on V0c;
"mean" is the plain mean of the members. Subsets of 5 and 10 members are random subsets (8 draws, mean of the draws).

**Menu v1** (the reference, `outputs/menu_retrain_v0/sc31_{10,30}k_genomic`): 20 LLM-written architectures (`albench/ensemble/menus/menu_v1.json`), Muon, lr 2e-3,
weight decay 1e-4, batch 256, one-cycle, 60 to 80 epochs, patience 16, shift and reverse-complement augmentation, one training seed per architecture; validated
during training on `chr_val_ref_only.npz` (the same sequences as V0c with different oracle labels, r = 0.983), so its stack is re-fitted on V0c here.

**MPRA-LegNet** (`scripts/program_oct22/bench33_legnet.py`): the published architecture (`external/human_legnet/model.py`: stem 64 / ks 11, four EffBlocks ks 9, widths
80 96 112 128, MaxPool 2 each, resize factor 4, SiLU, 1.32M parameters) trained from scratch with the published optimiser (AdamW, weight decay 0.1, OneCycleLR max_lr
0.01 from max_lr/25, pct_start 0.3, MSE, fp16 autocast, gradient clipping 1, best epoch by V0c Pearson). Shift (uniform -10..10 on the real plasmid flank) and
reverse complement (p 0.5) augmentation. Three schedules: `pub` batch 1024 for 25 epochs (the published K562 values), `e100` batch 1024 for 100 epochs, `b256` batch 256
for 60 epochs; the one with the best mean single-model V0c Pearson is the "tuned" arm in the figure. 20 seeds per schedule and size.

**AlphaGenome-FT-MPRA** (`scripts/program_oct22/bench33_ag.py`, the recipe of `experiments/train_oracle_s2_v2.py` that trained the oracle members): AlphaGenome encoder
(all_folds weights) with the `boda-flatten-512-512` head (1 track, dropout 0.1), 600 bp input = 200 bp insert + 200 bp real plasmid flank each side. Stage 1: encoder
frozen, head only, AdamW lr 1e-3, weight decay 1e-6, batch 128, up to 60 epochs, patience 6, shift +-15 on the real flank (p 0.5) and reverse complement (p 0.5). Stage 2: the
Stage 1 checkpoint is written to disk, loaded back, and the job asserts that the loaded model reproduces the Stage 1 validation Pearson before any Stage 2 update
(`s2_start_val_after_loading_s1` in each meta file); then every encoder block is unfrozen. Recipe `oracle`: encoder lr 1e-4, head lr 1e-3, constant, up to 60 epochs, patience 4.
Recipe `winner` (memory `project_ag_s2_winner`): encoder lr 1.2e-4, head lr 5e-4, 3 head-only warm-up epochs, cosine decay, 60 epochs. Stage 1 checkpoints:
`outputs/bench33/s1_ckpt/{recipe}_{D}k_s{index}/best_model/checkpoint`. Hardware: H100 NVL (AlphaGenome stage 2 needs about 55 GB), one model per job.

## Results (Pearson r against oracle_v3)

### D = 30k

| arm | models | single (mean ± sd) genomic / OOD / random | ensemble, plain mean | ensemble, stacked on V0c | r vs experiment, genomic test (single / ensemble) |
|---|---|---|---|---|---|
| menu v1, 20 architectures | 20 | 0.745±0.009 / 0.249±0.021 / 0.643±0.012 | 0.780 / 0.280 / 0.711 | 0.782 / 0.271 / 0.711 | 0.708 / 0.742 |
| MPRA-LegNet, pub | 20 | 0.728±0.005 / 0.227±0.019 / 0.626±0.008 | 0.749 / 0.247 / 0.671 | 0.750 / 0.250 / 0.669 | 0.692 / 0.712 |
| MPRA-LegNet, e100 | 20 | 0.754±0.006 / 0.255±0.017 / 0.655±0.005 | 0.780 / 0.280 / 0.709 | 0.781 / 0.282 / 0.707 | 0.717 / 0.742 |
| MPRA-LegNet, b256 | 20 | 0.757±0.006 / 0.255±0.017 / 0.670±0.005 | 0.781 / 0.278 / 0.714 | 0.782 / 0.279 / 0.713 | 0.720 / 0.742 |
| AlphaGenome-FT-MPRA S2 from S1, oracle recipe | 6 | 0.961±0.000 / 0.803±0.009 / 0.970±0.000 | 0.963 / 0.806 / 0.972 | 0.963 / 0.806 / 0.972 | 0.916 / 0.917 |
| AlphaGenome-FT-MPRA S1 (frozen encoder), oracle | 6 | 0.938±0.000 / 0.777±0.009 / 0.939±0.001 | 0.941 / 0.784 / 0.942 | 0.941 / 0.783 / 0.942 | 0.894 / 0.897 |
| AlphaGenome-FT-MPRA S2 from S1, winner recipe | 1 | 0.963±0.000 / 0.802±0.000 / 0.972±0.000 | 0.963 / 0.802 / 0.972 | - | 0.917 / 0.917 |
| AlphaGenome-FT-MPRA S1 (frozen encoder), winner | 1 | 0.938±0.000 / 0.778±0.000 / 0.934±0.000 | 0.938 / 0.778 / 0.934 | - | 0.894 / 0.894 |

### D = 10k

| arm | models | single (mean ± sd) genomic / OOD / random | ensemble, plain mean | ensemble, stacked on V0c | r vs experiment, genomic test (single / ensemble) |
|---|---|---|---|---|---|
| menu v1, 20 architectures | 20 | 0.651±0.012 / 0.146±0.032 / 0.513±0.019 | 0.694 / 0.170 / 0.588 | 0.700 / 0.180 / 0.595 | 0.618 / 0.659 |
| MPRA-LegNet, pub | 20 | 0.611±0.017 / 0.070±0.021 / 0.443±0.027 | 0.640 / 0.080 / 0.494 | 0.647 / 0.090 / 0.509 | 0.581 / 0.608 |
| MPRA-LegNet, e100 | 20 | 0.624±0.015 / 0.117±0.028 / 0.462±0.024 | 0.660 / 0.135 / 0.524 | 0.664 / 0.150 / 0.534 | 0.593 / 0.627 |
| MPRA-LegNet, b256 | 20 | 0.642±0.012 / 0.121±0.037 / 0.488±0.022 | 0.674 / 0.142 / 0.553 | 0.679 / 0.163 / 0.550 | 0.610 / 0.640 |
| AlphaGenome-FT-MPRA S2 from S1, oracle recipe | 5 | 0.950±0.001 / 0.782±0.007 / 0.956±0.001 | 0.951 / 0.786 / 0.957 | 0.951 / 0.784 / 0.958 | 0.905 / 0.906 |
| AlphaGenome-FT-MPRA S1 (frozen encoder), oracle | 5 | 0.928±0.001 / 0.755±0.010 / 0.930±0.001 | 0.931 / 0.762 / 0.932 | 0.931 / 0.762 / 0.932 | 0.885 / 0.887 |
| AlphaGenome-FT-MPRA S2 from S1, winner recipe | 1 | 0.951±0.000 / 0.787±0.000 / 0.958±0.000 | 0.951 / 0.787 / 0.958 | - | 0.906 / 0.906 |
| AlphaGenome-FT-MPRA S1 (frozen encoder), winner | 1 | 0.930±0.000 / 0.763±0.000 / 0.931±0.000 | 0.930 / 0.763 / 0.931 | - | 0.886 / 0.886 |

