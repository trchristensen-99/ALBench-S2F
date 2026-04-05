# ALBench-S2F Experiment Tracker

> **Last updated:** 2026-04-05 afternoon
> **Purpose:** Track all experiments, hyperparameters, results, and gaps.

---

## Table of Contents
1. [Data Setup](#data-setup)
2. [Model Architectures](#model-architectures)
3. [Experiment 0: Scaling / Random Subsampling](#exp0)
4. [Bar/Scatter Plot: MPRA Model Comparison](#barplot)
5. [Technique Sweep Results](#techniques)
6. [Oracle Ensembles](#oracles)
7. [Known Issues & Bugs](#bugs)
8. [Gap List / TODO](#gaps)

---

## 1. Data Setup <a name="data-setup"></a>

### K562 MPRA Dataset
- **Source:** Gosai et al. 2024 Nature, Zenodo
- **File:** `data/k562/DATA-Table_S2__MPRA_dataset.txt`
- **Total oligos:** ~798K (ref + alt alleles), ~401K (ref-only)
- **Sequence length:** 200bp variable inserts, padded to 600bp with MPRA vector flanks
- **Cell types:** K562, HepG2, SK-N-SH (3 label columns)
- **Quality filter (added 2026-04-02):** project filter (UKBB/GTEX/CRE), max SE < 1.0, ±6σ outlier removal (+4 upper shift)
- **After quality filter:** ref+alt ~738K (from 798K), ref-only ~401K
- **NOTE:** All runs before 2026-04-02 evening used UNFILTERED data

### Splits

| Split | Method | ref-only | ref+alt | Test chromosomes |
|---|---|---|---|---|
| **HashFrag** | Sequence homology (SW score < 60) | train=320K, val=41K, test=41K | Same 401K (cache is ref-only) | N/A (random) |
| **Chr-based** | Chromosome holdout | train=337K, val=31K, test=33K | train=659K, val=61K, test=66K | test=chr7+13, val=chr19+21+X |

### Yeast Dataset
- **Source:** Vaishnav et al. 2022
- **Total:** ~6M sequences, 80bp
- **Splits:** random 80/10/10

---

## 2. Model Architectures <a name="model-architectures"></a>

### From-Scratch Models

| Model | Params | Architecture | Input | Key HPs |
|---|---|---|---|---|
| **LegNet** | ~2.65M | 8 EffBlocks (SE + ResidualConcat), k=5, block_sizes=[256,256,128,128,64,64,32,32] | (B, C, L) C=5 K562, C=4 yeast | lr=0.005, bs=1024, wd=0.01, dropout=0.2 |
| **DREAM-CNN** | ~1.94M | BHI dual-kernel stem (k=9,15) + 5 inverted residual blocks with SE | (B, 4, L) | lr=0.005, bs=512, wd=0.01, dropout=0.2 |
| **DREAM-RNN** | ~1.2M | Bidirectional LSTM (hidden=256) + linear head | (B, 5, L) with RC flag | lr=0.005, bs=128, wd=0.01 |
| **Malinois (our default)** | ~5.3M | Basset: 3 conv, 2 linear, 1 branched (250ch), MSE loss | (B, 4, 600) padded | lr=0.00327, bs=512, wd=3.44e-4 |
| **Malinois (paper-mode)** | ~4.1M | Basset: 3 conv, 1 linear, 3 branched (140ch, dp=0.576), L1KL loss (α=1,β=5), multi-task 3 outputs, Adam(betas=0.866/0.879, amsgrad), Basset pretrained conv weights, RC interleave, dup cutoff=0.5 | (B, 4, 600) padded | Same LR/WD |

### Foundation Models (Encoder + Head)

| Model | Encoder Params | Head | Embed Dim | S1 (frozen) | S2 (fine-tuned) |
|---|---|---|---|---|---|
| **AlphaGenome** | ~550M | boda-flatten-512-512 MLP | 1536 | head-only, cached embeddings | unfreeze blocks 4+5 (K562) or all (yeast) |
| **Enformer** | ~251M | MLP (512 hidden, LayerNorm) | 3072 | head-only, cached embeddings | unfreeze all transformer layers |

---

## 3. Experiment 0: Scaling / Random Subsampling <a name="exp0"></a>

**Setup:** HashFrag splits, ref-only (~400K pool), random reservoir sampling at 7 sizes (K562) or 10 sizes (yeast), 3 seeds per HP config.

### K562 Exp0 Completeness (updated 2026-04-04)

| Student | AG S1 oracle (def) | DRNN oracle | LegNet oracle | AG S2 oracle | Ground truth |
|---|---|---|---|---|---|
| AG S1 | COMPLETE (72) | COMPLETE (72) | **ln_oracle RUNNING** | **ag_s2_oracle RUNNING** | N/A |
| AG S2 warm | 21 (3/size) | — | pending | pending | N/A |
| DREAM-RNN | 30 (gaps@4) | 36 (gaps@2) | pending | pending | 22 (gaps@2 large) |
| DREAM-CNN | 31 (gaps@5) | 36 (gaps@2) | pending | pending | 26 (gaps@2 large) |
| LegNet | 66 (gaps@2) | 66 (gaps@2) | pending | pending | 75 (gaps@2 large) |

**Oracle label status:**
- AG S1 oracle: ✅ COMPLETE (10-fold, `ag_hashfrag_oracle_cached/oracle_{0-9}`)
- DREAM-RNN oracle: ✅ COMPLETE (10-fold, `oracle_dream_rnn_k562_ensemble/oracle_{0-9}`)
- LegNet oracle: **RUNNING** (10-fold, `oracle_legnet_k562_ensemble/`)
- AG S2 oracle: **RUNNING** (10-fold S2 fine-tuning, `stage2_k562_oracle/fold_{0-9}`)

**HP configs used:**
- LegNet: lr=[0.001, 0.003, 0.005, 0.01], bs=[512, 1024] — best: **lr=0.005-0.01, bs=512** (HP probe confirmed)
- DREAM-CNN: lr=[0.001, 0.005, 0.01], bs=[512, 1024] — best: **lr=0.005, bs=512** (HP probe: lr=0.01 ≈ same, lr=0.001 worse)
- DREAM-RNN: lr=[0.005], bs=[128, 512]
- AG S1: lr=[3e-4, 1e-3], bs=[128, 256]

**Result dirs:** `outputs/exp0_oracle_scaling_v4/k562/{model}/random/n{size}/hp{idx}/seed{seed}/result.json`

### Yeast Exp0 Completeness

| Student | DRNN oracle (default) | AG oracle | Ground truth | Status |
|---|---|---|---|---|
| LegNet | 48 results (4/10) | 44 results (3/10) | 57 results (7/10) | **RUNNING** |
| DREAM-CNN | 40 results ✅ | 40 results ✅ | 28 results (3/10) | GT **RUNNING** |
| DREAM-RNN | 41 results ✅ | 42 results ✅ | 25 results (2/10) | GT **RUNNING** |
| AG S1 | 84 results ✅ | 84 results ✅ | — | |
| AG S2 cold | 86 results (9/10) | 83 results ✅ | — | Missing n=6M |
| AG S2 warm | 0/10 | — | — | **RUNNING** (shape fix applied) |
| AG S2 hlr | 30 results ✅ | — | — | lr=1e-3 variant |

**Result dirs:** `outputs/exp0_oracle_scaling_v4/yeast/{model}/random/n{size}/hp{idx}/seed{seed}/result.json`

### Plots
- `results/exp0_scaling_plots/k562_scaling_2x2.png` — all models, oracle labels
- `results/exp0_scaling_plots/yeast_scaling_2x2.png` — all models, oracle labels
- `results/exp0_scaling_plots/{task}_cross_oracle.png` — cross-oracle comparison
- `results/exp0_scaling_plots/{task}_oracle_vs_real.png` — oracle vs real labels
- **Script:** `scripts/analysis/plot_exp0_scaling_curves.py`

---

## 4. Bar/Scatter Plot: MPRA Model Comparison <a name="barplot"></a>

**Setup:** Chr-based splits, ref+alt (~786K), weighted average across K562/HepG2/SknSh, OOD = K562-only (synthetic seqs only have K562 labels).

### Chr_split Results (v2 = ref+alt preferred)

| Model | K562 | HepG2 | SknSh | Predictions? |
|---|---|---|---|---|
| Malinois | 3 seeds (v2) ✅ | 3 seeds (v2) ✅ | 1 seed (v2) + 2 **RUNNING** | No |
| LegNet | 1 seed done (0.812) + 1 running | 2 seeds **PENDING** (slow) | 2 seeds **PENDING** (slow) | No |
| DREAM-RNN | 1 seed (v2) + 1 **RUNNING** | 2 seeds **RUNNING** | 2 seeds **RUNNING** | No |
| Enf. S1 | 3 seeds (v1 ref-only) | 3 seeds (v1) | 3 seeds (v1) | No |
| Enf. S1 v2 | **REBUILDING** cache | **REBUILDING** | **REBUILDING** | No |
| Enf. S2 | **MISSING** (needs S1) | **MISSING** | **MISSING** | No |
| AG S1 | 1 seed (v2) ✅ | 1 seed (v2) ✅ | 1 seed (v2) ✅ | No |
| AG S2 warm | 1 seed ✅ (0.875) | **RUNNING** (default) | **RUNNING** (default) | No |
| AG S2 cold | 1 seed (v1 ref-only) | 1 (v1) | 1 (v1) | No |

**Result dirs:** `outputs/chr_split_v2/{cell}/{model}/...`
**Fallback dirs:** `outputs/chr_split/{cell}/{model}/...` (old v1 ref-only)

### Key Bar Plot Numbers (current)

| Model | Reference | SNV Effect | Synthetic Seqs |
|---|---|---|---|
| Malinois | 0.830 | 0.337 | 0.500 |
| DREAM-RNN | 0.836 | 0.305 | 0.404 |
| Enf. (Probing) | 0.862 | 0.295 | 0.331 |
| AG (Probing) | 0.878 | 0.355 | 0.707 |
| AG (Fine-tuned) cold | 0.869 (cold!) | 0.343 | 0.641 |
| AG (Fine-tuned) warm blocks[4,5] | 0.875 | 0.337 | 0.668 |
| **AG (Fine-tuned) FIXED RC+shift** | **0.895** | **0.347** | **0.697** |

### Colors (from PI meeting notes)
- Malinois/LegNet: `#E8DCCF` (baseline beige) / `#D4A017` (gold)
- DREAM-RNN: `#8B9DAF` (blue-gray)
- Enf. Probing: `#E7CDC2`, Fine-tuned: `#A65141`
- AG Probing: `#80A0C7`, Fine-tuned: `#394165`

### Plots
- `results/alan_style_plots/mpra_benchmark_chr_split.png`
- **Script:** `scripts/analysis/plot_alan_style_barplot.py`

---

## 5. Technique Sweep Results <a name="techniques"></a>

All on K562 chr_split, ref+alt.

### Pre-quality-filter results (unfiltered data)

| Technique | Malinois | DREAM-RNN | AG S1 |
|---|---|---|---|
| **Baseline (RC only)** | 0.835 | 0.822 | 0.881 |
| **+ Shift (±15bp)** | **0.839** (+0.4%) | **0.830** (+0.8%) | — |
| **+ High-activity dup** | 0.838 (+0.3%) | **0.852** (+3.0%) | 0.884 (+0.3%) |
| + Shift + dup | — | 0.817 (worse!) | — |
| + RC interleave | 0.836 (≈0) | — | — |
| + Cosine LR | 0.838 (≈0) | — | — |
| No augmentation | 0.827 (-0.8%) | — | — |

### Quality-filtered results (boda2 preprocessing, 2026-04-02)

| Technique | Malinois | LegNet | AG S1 | AG S2 (20K) |
|---|---|---|---|---|
| **Baseline (QF + ref+alt + RC)** | 0.847 (+1.2%) | 0.837 | **0.902** (+2.1%) | — |
| **+ Shift (±15bp)** | 0.851 | **0.797 (HURTS!)** | — | — |
| **+ Dup (cutoff=0.5)** | 0.850 | 0.831 | 0.901 | — |
| **+ Shift + Dup** | **0.858** | pending | — | — |
| AG S2 all-blocks 20K | — | — | 0.883 (S1@20K) | 0.853 (still < S1) |

**Key findings:**
1. **Quality filter = +1-2% for ALL models** (single biggest improvement)
2. **Malinois shift+dup = 0.858** (best from-scratch with our default architecture)
3. **LegNet shift is HARMFUL** (-4%): k=5 kernel too sensitive to positional shifts
4. **AG S1 at 0.902** with quality filter — near paper level
5. **AG S2 FIXED (2026-04-02): 0.895 with RC+shift** — 5 fixes applied
6. **LegNet shift ALWAYS hurts** — even ±3bp (-3.1%). Use baseline only.

### Malinois Paper-Mode Investigation (2026-04-04)

**Paper-mode (correct boda2 arch+loss+optimizer) underperforms pretrained model:**
- Paper-mode K562 in_dist: **0.840-0.849** (3 seeds)
- Pretrained Malinois: **0.883** (chr-split)
- Our default architecture (MSE, 2 linear, wider): **0.858**

**Root causes of gap (verified from pretrained checkpoint metadata):**
1. **Different training data**: pretrained used private `MPRA_ALL_v3.txt` including **BODA synthetic CREs** — we only have public `Table_S2__MPRA_dataset.txt` without BODA project
2. **Batch size**: pretrained used 1076 (fixed in paper_v2), our default was 512
3. **Scheduler T_0**: pretrained used 4096 steps (fixed in paper_v2), ours scaled with loader
4. **Early stopping**: pretrained used custom `entropy_spearman`, we use `val_pearson`
5. **Outlier filter**: pretrained used `+3.0` upper shift, our code had `+4.0`

**Conclusion**: Full replication from public data is NOT possible because the pretrained model was trained on data we don't have access to. The paper_v2 run (with fixed batch_size=1076 and T_0=4096) is **RUNNING** and should close part of the gap.

**Result dirs:** `outputs/aug_sweep/`, `outputs/techniques_sweep/`, `outputs/multitask/`

---

## 6. Oracle Ensembles <a name="oracles"></a>

| Oracle | Location | Type | Status |
|---|---|---|---|
| AG S1 K562 (10-fold) | `outputs/ag_hashfrag_oracle_cached/oracle_{0-9}` | k-fold head-only | ✅ COMPLETE |
| AG S2 K562 (10-fold) | `outputs/stage2_k562_oracle/fold_{0-9}` | k-fold fine-tuned encoder | ✅ COMPLETE (15-20 min/fold) |
| AG S2 K562 pseudolabels | `outputs/oracle_pseudolabels_stage2_k562_ag/` | ensemble pseudo-labels | **GENERATING** (ag_s2_pseudo) |
| DREAM-RNN K562 | `outputs/oracle_dream_rnn_k562_ensemble/` | 10 oracle folds | ✅ COMPLETE |
| LegNet K562 (10-fold) | `outputs/oracle_legnet_k562_ensemble/` | k-fold, 10 folds | ✅ COMPLETE |
| DREAM-RNN Yeast | `outputs/oracle_dream_rnn_yeast_kfold_v2/` | k-fold | ✅ COMPLETE |
| AG Yeast | `outputs/oracle_alphagenome_yeast_ensemble/` | k-fold | ✅ COMPLETE |

---

## 7. Known Issues & Bugs <a name="bugs"></a>

| Issue | Status | Impact | Details |
|---|---|---|---|
| AG S2 cold start | **FIXED** | S2 < S1 on real labels | Was reinitializing head. Now copies S1 head weights. |
| AG S2 dropout | **WORKAROUND** | Small (~0.1-0.5%) | alphagenome_ft doesn't pass is_training; editable install caches old code |
| OOD label mismatch | **FIXED** | HepG2/SknSh OOD was ~0 | OOD seqs only have K562 labels; now K562-only for OOD |
| Foundation S1 chr_split test | **FIXED** | Wrong test set | Was using hashfrag test, now uses chr7+13 |
| HashFrag include_alt_alleles | **KNOWN** | No effect | HashFrag cache was built ref-only; only chr-based supports ref+alt |
| orbax checkpoint save | **FIXED** | S1 checkpoints not saving | Needed absolute paths + parent-only mkdir |
| Yeast S2 warm shape mismatch | **FIXED** | Output layer 1 vs 18 | Now skips layers with mismatched shapes |

---

## 8. Gap List / TODO <a name="gaps"></a>

> Updated 2026-04-04

### CRITICAL BUG FIXED (2026-04-04): Malinois Architecture Mismatch
Our Malinois implementation had MAJOR differences from the paper (boda2):
- Wrong loss (MSE vs L1KL), wrong arch (2 linear vs 1, 1 branched vs 3),
  wrong optimizer (default Adam vs tuned betas+amsgrad), no transfer learning.
- `paper_mode=True` now matches boda2 exactly. All Malinois results are being re-run.
- Pretrained weights downloaded from GCS (49MB model + 19MB Basset conv).
- **Pretrained Malinois** chr-split in_dist: K562=0.8833, HepG2=0.8873, SknSh=0.8781

### Architecture Verification (2026-04-04)
| Model | Status | Notes |
|---|---|---|
| Malinois | **WAS BROKEN → FIXED** | paper_mode matches boda2 exactly |
| DREAM-RNN | Minor diff | Final head has extra capacity vs reference |
| LegNet | Correct (original defaults) | Optimized config uses ks=7, different blocks |
| DREAM-CNN | Intentional hybrid | BHI stem + smaller LegNet core |
| Enformer S1 | Correct | Standard trunk embeddings, center 4 bins |

### Jobs Pipeline (2026-04-05 afternoon)

**Running (13 jobs across all 3 QoS tiers):**

| Job | QoS | Task | Time |
|---|---|---|---|
| dcnn_k562 | **fast** | DREAM-CNN K562 3 seeds | just started |
| dcnn_other | **default** | DREAM-CNN HepG2+SknSh 3 seeds each | just started |
| mal_ablat | **default** | Malinois paper shift + nodup ablations | just started |
| exp0_k562_dream_gaps | slow_nice | K562 DREAM gaps at medium fracs | 10h |
| bar_3rdseed | slow_nice | 3rd seeds + HepG2/SknSh ensembles | just started |
| gen_preds | slow_nice | Predictions.npz from all existing checkpoints | 1min |
| ag_s2_pseudo | slow_nice | AG S2 pseudolabel generation (folds done) | 11min |
| ag_seeds2 | slow_nice | AG S1/S2 extra seeds, all cells | 11min |
| exp0_legnet_oracle | slow_nice | All students × LegNet oracle K562 | 10h |
| exp0_large | slow_nice | K562 large-frac gaps | 23h |
| exp0_yeast_large | slow_nice | Yeast large-frac gaps | 23h |
| exp0_legnet | slow_nice | LegNet yeast scaling | 46h |

**Pending (2, chained):**

| Job | Depends on | Task |
|---|---|---|
| exp0_ag_s2_o | ag_s2_pseudo | All students × AG S2 oracle K562 |

### Completed (2026-04-04–05)

| Job | Result |
|---|---|
| AG S2 oracle 10-fold | ✅ All 10 folds (15-20 min each) |
| LegNet oracle 10-fold | ✅ All 10 folds (6h39m total) |
| Malinois paper v2 (batch=1076, T_0=4096) | K562 in_dist = **0.850** |
| Malinois paper (pretrained, 3 seeds) | K562 in_dist = 0.840-0.849 |
| Malinois paper (nopretrain, 3 seeds) | K562 in_dist = 0.842-0.871 |
| Malinois paper (chr_split ref-only, 6 runs) | ✅ DONE |
| Yeast S2 warm n=6M | ✅ DONE |
| Malinois pretrained eval | K562=0.883, HepG2=0.887, SknSh=0.878 |
| Enformer S1 K562 (bar_final) | val=0.894, 1 seed |

### Systematic Comparison: Malinois Paper Techniques

| Condition | Splits/Data | K562 in_dist | Status |
|---|---|---|---|
| paper-mode + pretrained (3 seeds) | chr-split ref+alt | 0.840-0.849 | ✅ DONE |
| paper-mode, no pretrained (3 seeds) | chr-split ref+alt | 0.842-0.871 | ✅ DONE |
| paper v2 (bs=1076, T_0=4096, 3 seeds) | chr-split ref+alt | **0.848-0.850** | ✅ DONE |
| paper-mode + pretrained (3 seeds) | chr-split ref-only | 0.848-0.878 | ✅ DONE |
| paper-mode, no pretrained (3 seeds) | chr-split ref-only | 0.871-0.876 | ✅ DONE |
| paper-mode + pretrained + shift (3 seeds) | chr-split ref+alt | TBD | **RUNNING** (mal_ablat) |
| paper-mode + pretrained, no dup (3 seeds) | chr-split ref+alt | TBD | **RUNNING** (mal_ablat) |
| HashFrag conditions | — | — | **SKIPPED** (BLAST+ unavailable) |
| Pretrained model eval (no training) | chr-split | **0.883** | ✅ DONE |

**Malinois performance gap explained:** The pretrained model was trained on a private data file (`MPRA_ALL_v3.txt`) including BODA synthetic CREs not in the public dataset. Our best from public data: **0.850** (paper_v2). Our default architecture (MSE, 2 linear): **0.858**. Gap is primarily from different training data, not settings.

### Bar Plot Results (2026-04-05)

**bar_final ref+alt (outputs/bar_final/):**

| Model | K562 | HepG2 | SknSh | Seeds | Status |
|---|---|---|---|---|---|
| AG S1 | 0.902 | 0.902 | 0.891 | 1→3 | **ag_seeds2 RUNNING** |
| AG S2 (rc+shift) | 0.895 | 0.901 | 0.890 | 1→3 | **ag_seeds2 RUNNING** |
| DREAM-RNN | 0.879/0.843 | ~0.87 | ~0.87 | 2→3 | **bar_3rdseed RUNNING** |
| DREAM-RNN ens3 | 0.890 | — | — | 1 | **bar_3rdseed RUNNING** |
| LegNet | 0.840/0.836 | ~0.84 | ~0.83 | 2→3 | **bar_3rdseed RUNNING** |
| LegNet ens3 | 0.855 | — | — | 1 | **bar_3rdseed RUNNING** |
| DREAM-CNN | — | — | — | 0→3 | **dcnn_k562 + dcnn_other RUNNING** |
| Malinois (paper, pretrained) | 0.840-0.849 | — | — | 3 | ✅ DONE |
| Malinois (paper v2) | 0.848-0.850 | — | — | 3 | ✅ DONE |
| Malinois (our arch) | 0.858 | 0.854 | 0.848 | 3 | ✅ DONE |
| Enformer S1 | 0.894 (K562 only) | — | — | 1 | K562 done, HepG2/SknSh use chr_split |
| Enformer S2 | 0.883 | 0.850 | 0.853 | 3 | In separate dirs (usable) |
| Pretrained Malinois | 0.883 | 0.887 | 0.878 | N/A | Eval only |

**chr_split ref-only (outputs/chr_split/) — COMPLETE for most models:**
AG S1/S2, DREAM-RNN (3), DREAM-CNN (3), Enformer S1 (3), Borzoi S1 (3), Malinois (3+6 paper), NTv3 (3) — all 3 cells.

### Exp0 K562 Completeness (2026-04-05)

| Student | AG S1 oracle | DRNN oracle | LegNet oracle | AG S2 oracle | Ground truth |
|---|---|---|---|---|---|
| AG S1 | ✅ 72 | ✅ 72 | **RUNNING** | **PENDING** | N/A |
| AG S2 | 21 (3/size) | — | **RUNNING** | **PENDING** | N/A |
| DREAM-RNN | 30 (gaps filling) | 36 (gaps filling) | **RUNNING** | **PENDING** | 22 (gaps filling) |
| DREAM-CNN | 31 (gaps filling) | 36 (gaps filling) | **RUNNING** | **PENDING** | 26 (gaps filling) |
| LegNet | 66 (gaps filling) | 66 (gaps filling) | **RUNNING** | **PENDING** | 75 (gaps filling) |

### Exp0 Yeast Completeness
All models incomplete at large fractions (n>=1.2M). exp0_legnet (46h) and exp0_yeast_large (23h) filling these.

### Predictions for Scatter Plots
**gen_preds RUNNING** — generating predictions.npz from all existing Malinois + Enformer checkpoints.
New training runs (dcnn, bar_3rdseed, ag_seeds2) save predictions automatically.

### Remaining After Current Jobs

1. ✅ All oracle labels generated (AG S1, AG S2, DREAM-RNN, LegNet for K562; AG, DREAM-RNN for yeast)
2. **AG S2 oracle Exp0** — chained to ag_s2_pseudo, auto-starts when pseudolabels ready
3. **Fresh backup** — after all jobs complete
4. **Plot regeneration** — after results are final
5. **Enformer HepG2/SknSh** for bar_final — skipped (use chr_split ref-only results instead; cache building too expensive/slow)
6. **Yeast LegNet oracle** — lower priority (no existing K562-style oracle pipeline for yeast LegNet)
