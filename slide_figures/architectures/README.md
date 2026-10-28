# Menu-v1 architecture diagrams

Produced by `scripts/analysis/make_architecture_diagrams.py` (local, CPU; no cluster access).

## How the diagrams are made

Each menu entry's `source` is built with the repository builder (`albench.arch_search.validate.build_and_check`, forward-time gate disabled), its parameter count is compared with the `n_params` stored in `menu_v1.json`, and one forward pass on a (1, 4, 200) tensor is recorded with forward hooks on every module plus a `TorchFunctionMode` that captures functional ops (pooling, concat, gating, add) and which tensor each layer consumed. The boxes, the curved skip arrows and the tensor shapes come from that recording (helper-method names such as `composition()` come from the Python call stack), not from reading the code by hand. Runs of identical blocks are collapsed to `N× block`; a box shows the layers inside it in execution order. The panel at right opens the box with the most parameters (a repeated block, or a composite module).

Colour code: blue convolution, grey norm/activation, orange attention, purple recurrent, green pooling/reduction, gold add/concat/gate, dark linear head. Red curved arrows mark a tensor used by a non-adjacent layer (skip, branch, FiLM-style conditioning).

## Files

- `arch_son55_wide_rep2_r08_arch_15.png`: rank 1: son55_wide/rep2, 1,524,038 params, 30k weight 0.216
- `arch_son55_wide_rep1_r09_arch_19.png`: rank 2: son55_wide/rep1, 1,124,097 params, 30k weight 0.178
- `arch_evoaug_paper2023_rep1_r02_arch_03.png`: rank 3: evoaug_paper2023/rep1, 2,500,734 params, 30k weight 0.177
- `arch_motif_ct_enriched_rep1_r04_arch_08.png`: rank 4: motif_ct_enriched/rep1, 670,321 params, 30k weight 0.125
- `arch_random_rep3_r03_arch_06.png`: rank 5: random/rep3, 566,403 params, 30k weight 0.111
- `arch_motif_ct_enriched_rep3_r05_arch_10.png`: rank 6: motif_ct_enriched/rep3, 357,097 params, 30k weight 0.096
- `arch_evo2_rep3_r04_arch_08.png`: rank 7: evo2/rep3, 1,468,466 params, 30k weight 0.092
- `arch_motif_ct_enriched_rep3_r04_arch_08.png`: rank 8: motif_ct_enriched/rep3, 1,421,485 params, 30k weight 0.062
- `arch_evo2_rep2_r01_arch_02.png`: rank 9: evo2/rep2, 1,556,657 params, 30k weight 0.061
- `arch_random_rep1_r09_arch_18.png`: rank 10: random/rep1, 837,249 params, 30k weight 0.044
- `arch_GRM_30k_rep1_r06_arch_13.png`: rank 11: GRM_30k/rep1, 1,348,289 params, 30k weight 0.033
- `arch_GRM_30k_rep3_r01_arch_02.png`: rank 12: GRM_30k/rep3, 1,540,049 params, 30k weight 0.006
- `arch_motif_ct_enriched_rep3_r02_arch_04.png`: rank 13: motif_ct_enriched/rep3, 1,621,009 params, 30k weight 0.000
- `arch_motif_ct_enriched_rep2_r09_arch_17.png`: rank 14: motif_ct_enriched/rep2, 1,169,105 params, 30k weight 0.000
- `arch_GRM_30k_rep2_r05_arch_10.png`: rank 15: GRM_30k/rep2, 1,629,561 params, 30k weight 0.000
- `arch_evoaug_paper2023_rep3_r04_arch_09.png`: rank 16: evoaug_paper2023/rep3, 287,177 params, 30k weight 0.000
- `arch_evo2_rep1_r06_arch_12.png`: rank 17: evo2/rep1, 1,259,921 params, 30k weight 0.000
- `arch_evoaug_paper2023_rep2_r07_arch_13.png`: rank 18: evoaug_paper2023/rep2, 1,620,849 params, 30k weight 0.000
- `arch_son55_wide_rep3_r06_arch_09.png`: rank 19: son55_wide/rep3, 761,921 params, 30k weight 0.000
- `arch_random_rep2_r01_arch_02.png`: rank 20: random/rep2, 1,529,777 params, 30k weight 0.000
- `architectures_top6.png`: the six highest-weight entries as simplified diagrams in a 2 × 3 grid (one slide)
- `architectures_overview.png`: all 20 entries as small tiles: name, corpus, parameters, weight, block strip
- `menu_hyperparameters.png`: table of all 20: origin run, corpus, parameters, optimiser, lr, weight decay, batch size, epochs, schedule, search val r, 30k stack weight

## Parameter-count check (built network vs `n_params` in the menu)

All 20 built networks match the menu's `n_params` exactly.

## Entries

| rank | entry | params | 30k weight | structure found in the trace | diagram notes |
|---|---|---|---|---|---|
| 1 | `son55_wide_rep2_r08_arch_15` | 1,524,038 | 0.216 | attention, SE gating, residual add, skip concat | complete |
| 2 | `son55_wide_rep1_r09_arch_19` | 1,124,097 | 0.178 | SE gating, top-k pool, residual add, skip concat | complete |
| 3 | `evoaug_paper2023_rep1_r02_arch_03` | 2,500,734 | 0.177 | dilated, grouped conv, attention, SE gating, residual add | complete |
| 4 | `motif_ct_enriched_rep1_r04_arch_08` | 670,321 | 0.125 | grouped conv, SE gating, residual add | complete |
| 5 | `random_rep3_r03_arch_06` | 566,403 | 0.111 | grouped conv, attention, residual add | complete |
| 6 | `motif_ct_enriched_rep3_r05_arch_10` | 357,097 | 0.096 | dilated, grouped conv, 2-D pair map, SE gating, residual add | complete |
| 7 | `evo2_rep3_r04_arch_08` | 1,468,466 | 0.092 | U-Net, grouped conv, attention, SE gating, residual add, skip concat | complete |
| 8 | `motif_ct_enriched_rep3_r04_arch_08` | 1,421,485 | 0.062 | dilated, attention, SE gating, skip concat | complete |
| 9 | `evo2_rep2_r01_arch_02` | 1,556,657 | 0.061 | U-Net, dilated, grouped conv, residual add, skip concat | complete |
| 10 | `random_rep1_r09_arch_18` | 837,249 | 0.044 | dilated, grouped conv, SE gating, residual add | box 'mean, sub, bmm, div, abs …' is a run of 8 functional ops summarised by name (see source) |
| 11 | `GRM_30k_rep1_r06_arch_13` | 1,348,289 | 0.033 | dilated, grouped conv, attention, SE gating, top-k pool, residual add, skip concat | box 'max pool1d, ge, zeros like, where, top-…' is a run of 10 functional ops summarised by name (see source) |
| 12 | `GRM_30k_rep3_r01_arch_02` | 1,540,049 | 0.006 | U-Net, dilated, grouped conv, SE gating, residual add | complete |
| 13 | `motif_ct_enriched_rep3_r02_arch_04` | 1,621,009 | 0.000 | dilated, grouped conv, SE gating, residual add | complete |
| 14 | `motif_ct_enriched_rep2_r09_arch_17` | 1,169,105 | 0.000 | dilated, grouped conv, SE gating, top-k pool, residual add | complete |
| 15 | `GRM_30k_rep2_r05_arch_10` | 1,629,561 | 0.000 | grouped conv, SE gating, residual add, skip concat | box 'composition()' is a run of 19 functional ops summarised by name (see source) |
| 16 | `evoaug_paper2023_rep3_r04_arch_09` | 287,177 | 0.000 | dilated, grouped conv, SE gating, shared weights, residual add | complete |
| 17 | `evo2_rep1_r06_arch_12` | 1,259,921 | 0.000 | grouped conv, SE gating, residual add | complete |
| 18 | `evoaug_paper2023_rep2_r07_arch_13` | 1,620,849 | 0.000 | SE gating, skip concat | complete |
| 19 | `son55_wide_rep3_r06_arch_09` | 761,921 | 0.000 | dilated, skip concat | complete |
| 20 | `random_rep2_r01_arch_02` | 1,529,777 | 0.000 | U-Net, grouped conv, SE gating, residual add | complete |

## Entries whose diagram is partial

- `random_rep1_r09_arch_18`: box 'mean, sub, bmm, div, abs …' is a run of 8 functional ops summarised by name (see source)
- `GRM_30k_rep1_r06_arch_13`: box 'max pool1d, ge, zeros like, where, top-…' is a run of 10 functional ops summarised by name (see source)
- `GRM_30k_rep2_r05_arch_10`: box 'composition()' is a run of 19 functional ops summarised by name (see source)

## Caveats

- 'corpus' / 'search corpus' is the `family` field of the menu entry: the training corpus of the AutoResearch run that produced the configuration, not an architecture family. Architecture tags are derived from the trace (e.g. U-Net = upsample + concat; SE gating = sigmoid-gated mean; attention = softmax with a weighted sum).
- Parameters held directly by the top-level `Net` (for example a shared grouped-conv weight used through `F.conv1d`) are in the network total but not in a box's parameter tag, which counts module parameters only.
- Tensor shapes drop the batch dimension. For the window model (random_rep3_r03_arch_06) the batch axis carries the 5 overlapping windows of each sequence, so shapes after the unfold are per window.
- Boxes are merged to fit one slide (cap of 13 boxes in the main diagram and 10 in the grid and tiles); a merged box is named after its parts, joined with `+`, and keeps all their parameters. The right-hand panel always shows one block at layer level.
- All 20 entries share one training recipe (muon, lr 0.002, wd 1e-4, batch 256, onecycle); only the epoch budget (60 / 70 / 80) differs, so what distinguishes the menu is the network source.
- Stack weights are those of the 30k genomic ensemble in `data_oct32.json` (`weights30k`); the 'epochs' figure in the table is the budget with the number actually trained at 30k in brackets.
