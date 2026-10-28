# Scaling-law curves in log space (Oct 28)

Stacked-ensemble error against training-set size D, on log-log axes, so a power law `err = a * D^-b` is a straight line and `b` is its slope.
Two error measures: `1 - Pearson r` and `MSE` (original label scale, against the oracle_v3 labels). Three evaluation panels: genomic test (chr7/13),
designed high-activity (OOD), random sequences (random_32k). GC-sweep and dinucleotide-shuffle results are in the CSV and JSON only.

## Files

| file | what |
|---|---|
| `scaling_1mr_loglog.png` | 1 - r vs D, all eight strategies at 10k and 30k (new sc31 cells), exponent in each legend entry |
| `scaling_mse_loglog.png` | the same for MSE |
| `scaling_core_three.png` | 1 - r vs D for genomic / random / motif CT-enriched over 10k-100k (filled, same pools) and the older 33-configuration menu out to 300k or 1M (hollow); thin line = power-law fit, legend gives b and R² |
| `scaling_core_three_mse.png` | the same for MSE |
| `scaling_exponents.png` | fitted exponent b for every series and panel, 1 - r (top row) and MSE (bottom row); hatched = two-point slope |
| `scaling_law_table.csv` | every point: series, strategy, source directory, D, eval set, n models stacked, r, 1 - r, MSE |
| `scaling_law_data.json` | the same plus every fit (a, b, R², sizes) |

Reproduce: `scripts/analysis/analyze_scaling_law.py compute` (cluster, cpuq; reads `outputs/menu_retrain_v0/*`) then `... plot --data scaling_law_data.json --out slide_figures/scaling`.

## Series

* **sc31** (hatched in the exponent figure): `sc31_{10k,30k}_<strategy>`, menu v1 up to 20 configurations, patience 16, nested seed-42 pools. Eight strategies, **two sizes only**,
  so each slope is a two-point slope with no straightness check.
* **v1** (filled markers): `v1_{10k,30k,100k}_{genomic,random,motif_ct_enriched}_r1`, menu v1, 20 configurations, patience 8, same replicate-1 pool at each size. Three sizes, R² reported.
* **v0** (hollow markers): older 33-configuration menu, first seed only. Genomic: nested pools at 10k/30k/100k/300k. Random and motif CT-enriched: ladder pools at 10k and 100k,
  independent 30k pool, nested 300k, anchor cells at 600k and 1M. Different pool draws and protocol from v1, so use it for the shape at large D, not to compare levels with v1.

Every point is a stack (positive ElasticNetCV on z-scored validation predictions) of the usable models in that directory; the three skip-list entries and any model with non-finite or exploded predictions are left out.

## Fitted exponents b (err ~ D^-b), least squares in log-log

1 - r, core three (R² in brackets):

| strategy | series | genomic test | designed OOD | random sequences |
|---|---|---|---|---|
| Gosai genomic | v1, 10k-100k | 0.243 (0.996) | 0.148 (1.000) | 0.276 (0.996) |
| Uniform random | v1, 10k-100k | 0.184 (0.983) | 0.221 (0.919) | 0.333 (1.000) |
| Motif CT-enriched | v1, 10k-100k | 0.222 (0.996) | 0.215 (0.982) | 0.305 (0.994) |
| Gosai genomic | v0, 10k-300k | 0.214 (0.989) | 0.157 (0.986) | 0.247 (0.997) |
| Uniform random | v0, 10k-1M | 0.152 (0.982) | 0.217 (0.987) | 0.255 (0.983) |
| Motif CT-enriched | v0, 10k-1M | 0.179 (0.987) | 0.229 (0.989) | 0.254 (0.992) |

MSE, same cells: genomic test 0.215 / 0.160 / 0.195 (v1 genomic / random / motif), OOD 0.419 / 0.454 / 0.452, random sequences 0.259 / 0.335 / 0.322. R² between 0.82 and 0.998, the low one is random OOD (below).

Two-point sc31 exponents (1 - r, genomic test / OOD / random sequences): genomic 0.29 / 0.11 / 0.30, random 0.24 / 0.17 / 0.37, motif CT-enriched 0.23 / 0.14 / 0.30,
EvoAug 0.26 / 0.14 / 0.34, Zoonomia 0.27 / 0.08 / 0.32; the full list is in `scaling_exponents.png`.

## How straight the lines are

* On 3-6 sizes the log-log lines are straight to R² >= 0.98 in 17 of the 18 core cases for 1 - r, so a power law is a fair summary over a decade or two. It is not exact: the local slope drifts.
  Genomic test, 1 - r, v0: Gosai genomic 0.25 (10k-30k), 0.23 (30k-100k), 0.15 (100k-300k); uniform random 0.23, 0.17, 0.12, 0.10, 0.14 across 10k-30k ... 600k-1M; motif CT-enriched 0.24, 0.19, 0.17, 0.13, 0.12.
  So extrapolating a 10k-100k slope to 1M overstates the gain on the genomic test set (the curves bend toward a floor). On designed OOD the slope goes the other way: random 0.11, 0.31, 0.20, 0.20, 0.17 and motif 0.14, 0.22, 0.30, 0.24, 0.19.
* The exception is random OOD at 10k-30k. MSE barely moves (v1 5.39 -> 5.04, sc31 5.35 -> 4.39, v0 5.36 -> 4.76) and then falls 2.6x to 100k (1.92), so 1 - r R² = 0.92 and MSE R² = 0.82 for that series. Read it as an elbow, not a power law.
* 1 - r and MSE give similar exponents on the genomic and random-sequence panels, but MSE falls roughly twice as fast as 1 - r on designed OOD (b 0.42-0.45 vs 0.15-0.22). Not diagnosed here; a likely reason is that the OOD error includes an amplitude/calibration component that shrinks faster with data than rank agreement does.

## Caveats

* One seed and one pool per point, no error bars. The single-run noise on r is about 0.004-0.008, which on 1 - r near 0.2 is 2-4% and puts roughly +-0.015 on a three-point slope over 10k-100k. Differences in b of 0.03 or less between strategies are not meaningful.
* sc31 30k cells that had not finished all 20 configurations when the data were read: mutagenesis_mr10 7 models, random_gc_sweep 7, motif_shared_core 13, zoonomia 16, motif_ct_enriched 19 (marked `*` in the figure legends). Their 30k point, and so their slope, will move when the rest finish. The 10k points are all 20.
* sc31 vs v1 differ in patience (16 vs 8) and pool: the same strategy at the same D can differ by up to 0.008 in r between the two series (genomic 10k 0.700 vs 0.705, 30k 0.782 vs 0.781; random 10k 0.687 vs 0.696, 30k 0.760 vs 0.763).
* The v0 hollow series mixes pool draws across sizes for random and motif (see Series), so part of its curvature could be pool-to-pool variation; the genomic v0 series is nested.
* sc31 at 100k was cut from scope for five strategies (not run), and 100k is available only through the v1 series for the core three. Only the core three have anything above 100k (v0 anchors 600k/1M; genomic stops at 300k).
