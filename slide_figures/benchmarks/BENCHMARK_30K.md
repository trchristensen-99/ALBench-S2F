# 30k Gosai genomic: numbers to benchmark another AutoResearch approach against (Oct 28)

Training set: the canonical 30k genomic pool (`gosai_2024_k562_30k_autoresearch/train_30k.npz`, labels oracle_v3). Test r is Pearson r on the shared test sets (genomic chr7/13, designed high-activity, random 32k) against
the oracle_v3 labels; "measured" is r against the experimental labels on the genomic test set (the oracle itself is at 0.97 there). Ensembles are positive ElasticNetCV stacks fitted on V0c validation. Files: `data_bench33.json`,
`data_best30k.json`, `data_best30k_pools.json`, `data_menu_vs_legnet.json`.

| approach | models | genomic | designed | random | genomic, measured |
|---|---|---|---|---|---|
| MPRA-LegNet, published recipe, one model (mean of 20 seeds) | 1 | 0.728 | 0.227 | 0.626 | 0.692 |
| MPRA-LegNet, published recipe, ensemble | 20 | 0.750 | 0.250 | 0.669 | 0.712 |
| MPRA-LegNet, tuned (batch 256, 60 epochs), one model (mean of 20; best by V0c 0.765) | 1 | 0.757 | 0.255 | 0.670 | 0.720 |
| MPRA-LegNet, tuned, ensemble | 20 | 0.782 | 0.279 | 0.713 | 0.742 |
| AlphaGenome-FT-MPRA Stage 1 only (frozen encoder), one model | 1 | 0.938 | 0.777 | 0.939 | 0.894 |
| AlphaGenome-FT-MPRA Stage 2 from Stage 1, one model (mean of 10 seeds) | 1 | 0.961 | 0.803 | 0.970 | 0.916 |
| AlphaGenome-FT-MPRA Stage 2 from Stage 1, ensemble | 10 | 0.963 | 0.806 | 0.972 | 0.917 |
| Ours: one architecture of the menu (mean of 20) | 1 | 0.745 | 0.249 | 0.644 | 0.708 |
| Ours: best single model by V0c validation, any run, same training pool | 1 | 0.760 | 0.275 | 0.677 | 0.723 |
| Ours: one AutoResearch run alone (92 runs: mean, sd 0.006, best run 0.780) | about 20 | 0.769 | 0.268 | 0.696 | |
| Ours: menu v1 ensemble (20 architectures x 1 seed) | 20 | 0.782 | 0.271 | 0.711 | 0.742 |
| Ours: ALL runs on this training set, ensemble (1,249 models from 28 run directories) | 1,249 | **0.797** | 0.302 | 0.742 | 0.757 |
| Ours: the same, best 20 by validation | 20 | 0.796 | 0.304 | 0.741 | 0.757 |

Pooling models trained on other 30k draws of the genomic reservoir (replicate pools r1 to r15; 4,022 models) gives 0.805 / 0.311 / 0.760 (measured 0.765), but those models have jointly seen far more than 30k distinct sequences, so it is not a 30k number.
Best single model by validation on a single replicate pool ranges 0.738 to 0.770 across the 15 pools: the draw of the 30k sequences moves a single model by about 0.01.

## Reading notes
* AlphaGenome numbers against the oracle are inflated: oracle_v3 is a 10-member ensemble of the same backbone fine-tuned on Gosai labels, so the student distils its own teacher (one frozen-encoder epoch already reaches 0.934 on validation). The measured column is the fair one: 0.917 against 0.742 for ours and LegNet.
* Menu and LegNet ensembles are not an exact tie: genomic 0.7824 against 0.7820 (95% bootstrap interval of the difference -0.0014 to +0.0024), designed set LegNet ahead by 0.008 (interval excludes 0), random sequences level; at 10k the menu is ahead by 0.021 on genomic. The two ensembles make almost the same errors (error correlation 0.97 at 30k) and stacking them together adds only +0.004.
* Test-set use: no model, ensemble size or selection above used a test set; validation (V0c) selects the best single and the ensemble weights, and a 50/50 split of V0c picked K = 20 for the pooled stack.
