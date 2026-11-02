# The tuned MPRA-LegNet: per-size recipe, what it beats, and how it was measured (Nov 1, 2026)

Short version: the published MPRA-LegNet recipe (AdamW, peak lr 0.01, batch 1024, one-cycle, weight decay 0.1, **100 epochs**) is badly under-trained on our data sizes. Training longer, with more weight decay
at small sizes, makes a plain LegNet the strongest student we have at 30k and 100k and a close second to the 20-configuration menu at 10k. This note is the measured recommendation; the experiments are
the `lr38_*` cells (`scripts/program_oct22/register_lr38*.py`, recipe configurations in `scripts/program_oct22/bench33_legnet.py`, table in `scripts/analysis/analyze_legnet_recipes.py`).

## The recipe per training size

Input standard unchanged (200 bp + 20 real flank bases per side, shift +-10, RC augmentation and RC-averaged prediction), canonical nested genomic pools, V0c validation for best-epoch selection, 20 seeds per size.

| size | epochs | weight decay | `bench33_legnet.py --cfg` | single-model genomic r | gain over e100 | 20-seed stack, genomic (oracle labels / measured labels) |
|---|---|---|---|---|---|---|
| 10k | 800 | 0.4 | `e800_wd4` | 0.755 | +0.066 (SE 0.005); OOD +0.099 | 0.7801 / 0.7421 |
| 30k | 500 | 0.2 | `e500_wd2` | 0.8745 | +0.024 (SE 0.001); OOD +0.047 | 0.8907 / 0.8487 |
| 100k | 500 | 0.1 (published) | `e500` | 0.9269 | +0.007 (SE 0.0003) | 0.9381 / 0.8953 |

Smaller data wants more epochs and more regularisation. Boundaries found: weight decay 0.4 is too strong at 30k (-0.005 to -0.009), weight decay 0.2 hurts at 100k (-0.003 to -0.007 genomic, about -0.03 OOD); at 30k 800 epochs is no better than 500; at 10k the gain was still rising at 800 epochs (not saturated, the 10k recipe is a lower bound on what is possible). Halving the learning rate or
doubling the batch was worse at every size. Controls: the published recipe with 20 seeds (`bench38_ln_e100_*`) scores 0.689 / 0.850 / 0.920 single and 0.731 / 0.868 / 0.931 as a stack at 10k / 30k / 100k.

## How many epochs, and is there early stopping? (Nov 2)

The bank trainer has **no early stopping**: it runs the whole one-cycle schedule and keeps the epoch with the best validation Pearson (V0c, forward-only). That epoch is always at the end of the schedule (mean best epoch 473 of 500 at 30k, 477 of 500 at 100k, 723 of 800 at 10k; the earliest in any seed is 91% of the way), because a one-cycle schedule only anneals in its last third. The recipes are tuned per size and all long: **10k 800 epochs (wd 0.4), 30k 500 (wd 0.2), 100k 500 (wd 0.1)**; in optimiser steps that is 8k, 14.6k and 49k (batch 1024), so the small sets need the most epochs.

Single-model genomic test r by schedule length (5 seeds per arm, the 20-seed arm in bold; same pools, V0c selection):

| size | e100 | e150 | e200 | e300 | e500 | e800 |
|---|---|---|---|---|---|---|
| 30k, wd 0.1 | 0.8502 (20 seeds) | 0.8601 | 0.8621 | 0.8658 | 0.8700 | not run |
| 30k, wd 0.2 | not run | 0.8654 | `lr39_` running | 0.8730 | **0.8748** (20) | 0.8732 |
| 100k, wd 0.1 | 0.9202 (20) | 0.9245 | 0.9254 | 0.9261 | **0.9269** (20) | not run |
| 10k, wd 0.1 | 0.6891 (20) | 0.7077 | 0.7141 | 0.7196 | 0.7360 | not run |
| 10k, wd 0.2 | not run | 0.7191 | not run | 0.7356 | 0.7468 | 0.7510 |
| 10k, wd 0.4 | not run | not run | `lr39_` running | `lr39_` running | 0.7508 | **0.7557** (20) |

So 200 epochs costs about -0.005 at 30k (interpolating the wd 0.2 row; -0.008 at wd 0.1), -0.0015 at 100k and about -0.02 at 10k; **300 epochs is within 0.002 of 500 at 30k and within 0.001 at 100k**, which is where the loop should aim when compute matters. At 10k the gain was still rising at 800.

Early stopping inside the full schedule is far worse: replaying the stored per-epoch validation curves of the 20-seed banks, the validation Pearson plateaus between 20% and 40% of the schedule (30k mean 0.8146 at 20%, 0.8141 at 40%, 0.8384 at 60%, 0.8816 at 80%, 0.8928 at 90%, 0.8948 at 100%) because the learning rate is still high, so any patience short enough to save time stops there:

| size (recipe) | patience 8 | patience 16 | patience 32 | patience 64 |
|---|---|---|---|---|
| 30k (e500, wd 0.2): median stop epoch / validation r given up | 44 / 0.133 | 114 / 0.059 | 141 / 0.054 | 500 / 0.008 (a fifth of seeds still stop, min epoch 174) |
| 100k (e500): | 67 / 0.044 | 85 / 0.042 | 116 / 0.040 | 153 / 0.037 |
| 10k (e800, wd 0.4): | 24 / 0.191 | 81 / 0.121 | 184 / 0.060 | 266 / 0.050 |

(`lr39_*_es16` measure this on the test sets: the same schedules with patience 16, 5 seeds.) A shorter schedule is the efficient alternative, not an early stop: it completes its annealing. This is also what the loop harness did to every long recipe until Nov 2 (patience 8 on the validation MSE), see `LEGNET_VS_LOOP_AND_NEXT_ARMS.md`.

## What it means for the per-size ensemble (genomic-test stack, oracle labels)

| size | tuned LegNet (20 seeds) | menu v2 (20 configs) | tuned + menu v2 | tuned + menu v2 + menu v2 at 3x epochs |
|---|---|---|---|---|
| 10k | 0.7801 | 0.7846 | 0.7916 | **0.7941** |
| 30k | **0.8907** | 0.8674 | 0.8908 | 0.8905 |
| 100k | **0.9381** | 0.9205 | 0.9380 | 0.9381 |

Use the tuned LegNet seeds alone at 30k and above; at 10k add the menu (it contributes diversity, +0.012 to +0.014). The menu members are not under-trained: tripling their epochs with early stopping
off changes the genomic r by +0.001 +- 0.004 (`ep38_v2x3`). Giving the menu members the LegNet recipe lowers each member by 0.02 to 0.07 (`tr38_*`), so the LegNet advantage is the architecture plus
the long AdamW schedule, not either alone.

## Consequences for the rest of the study

* Anything that used the published recipe as "the tuned LegNet" understated it: the in-run baseline LegNet in AutoResearch (best of e100 / b256), the LegNet-versus-menu-versus-AutoResearch comparisons, and
  the scaling reference. They should be redone against the per-size recipe above. AutoResearch arms measured at 30k (3 pools): the best arm (reference shown from the start) reached a 30-model stack of 0.854
  against 0.891 for the tuned LegNet bank.
* AlphaGenome students stay baselines only (AGENTS.md); the numbers here are students that learn only from the training set.

## The collaborator's architectures, in our hands

Five of his open-world architectures were re-implemented from his descriptions (`scripts/analysis/build_menu_pi.py`, widths fitted to his stated parameter counts). With the Muon recipe at 30k a single model scores
val 0.881, genomic 0.843, OOD 0.32, random 0.78 against his stated 0.885 / 0.852 to 0.855 / 0.31 to 0.33 / 0.77 (about 0.01 lower on genomic, within seed noise and the winner's curse of picking the best of ~200).
They are below the tuned LegNet at every size and best at 10k (single 0.76 against 0.69 for the published recipe, before tuning). If his test labels are the same oracle labels, his best ensemble (top-5, 0.8317)
is 0.06 below the tuned 30k stack; on measured labels the tuned stack is 0.849. His label basis is unconfirmed.

## Caveats

Five-seed arms were compared with a 20-seed control (unpaired, SEs above); the final per-size arms were then run to 20 seeds. The recipe was selected on validation-set behaviour of the sweeps and reported on test
sets that were never used for a choice, but the arms themselves were chosen from the same sweep, so the gain over the published recipe is slightly optimistic at the margin (the margins are 5 to 10 SE at 10k and 30k).
At 10k the best arm was the longest one tried; longer training may help further. 300k and larger sizes have not been tuned.

## Comparison with the collaborator's Nov 2 results (30k genomic)

His table (single networks and "ensembles" that put two or three full networks inside one model) against the tuned LegNet at 30k (20 seeds, same label sets: oracle "genomic r" and experimental "measured genomic r"):

| model | genomic r (oracle labels) | measured genomic r |
|---|---|---|
| his "tuned LegNet" baseline (1.3M params, 60 epochs) | 0.799 | 0.760 |
| published recipe, e100, our harness (single network, mean of 20) | 0.850 | 0.809 |
| his best single networks (C510, C370, 5.7M to 7.3M params) | 0.862 to 0.864 | 0.821 to 0.822 |
| **tuned LegNet single network (e500, wd 0.2; mean of 20, 1.3M params)** | **0.8748** | **0.8336** |
| his best two or three network models (C426, C514, C538) | 0.874 to 0.877 | 0.832 to 0.836 |
| tuned LegNet, plain average of 2 / 3 independent networks | 0.8829 / 0.8856 | 0.8413 / 0.8439 |
| tuned LegNet, 20-seed stack | 0.8907 | 0.8487 |

A single tuned LegNet is about 0.011 above his best single network and a plain average of two LegNets is above his best twin and triple models. His baseline is 0.05 below the published recipe in our harness, which is
what an under-trained reference (60 epochs) looks like. The gap between oracle and measured labels (0.041 for his best, 0.041 for ours) is the same, so the two harnesses appear to use equivalent test sets, but that is unconfirmed.
(`scripts/analysis/measured_singles.py`.)
