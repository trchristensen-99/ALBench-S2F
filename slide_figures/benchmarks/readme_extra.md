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
